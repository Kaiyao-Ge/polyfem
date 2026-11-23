#include "BlindUnitTerm.hpp"

#include <polyfem/solver/BilateralSegmentCertifier.hpp>
#include <polyfem/solver/FastTetValidity.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/candidates/candidates.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/potentials/barrier_potential.hpp>
#include <ipc/potentials/friction_potential.hpp>
#include <ipc/utils/eigen_ext.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace polyfem::solver
{
	namespace
	{
		constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
		constexpr int NL = 12;

		/// Installs the capture target of the local derivatives for one assembly
		struct CaptureScope
		{
			explicit CaptureScope(assembler::LocalCapture *c)
			{
				assembler::local_capture() = c;
			}
			~CaptureScope() { assembler::local_capture() = nullptr; }
		};
	} // namespace

	// ------------------------------------------------------------------------------------------------------------ setup
	BlindUnitTerm::BlindUnitTerm(NLProblem &problem,
								 const std::vector<std::shared_ptr<Form>> &forms,
								 const Eigen::VectorXd &mass_full,
								 const ipc::CollisionMesh &collision_mesh,
								 const Eigen::SparseMatrix<double> *displacement_map,
								 const std::shared_ptr<const FastTetValidity> &fast_validity,
								 const Options &opts,
								 const std::shared_ptr<BlindSession> &session,
								 const std::shared_ptr<const LocalMass> &mass,
								 const bool full_size)
		: BarrierContactNonsmoothTerm(problem, forms, mass_full, collision_mesh, displacement_map, fast_validity, [&] {
			  Options o = opts;
			  o.first_layer = false; // the first layer of the blind selection is the one below
			  return o;
		  }()),
		  full_size_(full_size), session_(session), port_(*this), sink_(*this)
	{
		if (full_size_)
		{
			// the augmented Lagrangian subproblems: the unknowns are all full dofs (no reduction); the collision vertices (obstacle
			// included) map to full dofs through the same node map
			const int nf = problem_.full_size();
			red_of_full_.resize(nf);
			for (int i = 0; i < nf; ++i)
				red_of_full_[i] = i;
			local_ok_ = opts_.local_blocks && init_node_of_vertex(nf);
		}
		if (!local_ok_)
			log_and_throw_error("NonsmoothNewton: selection = blind needs the local contact blocks (plain selection dof map, one FE node per collision vertex, debug/local_blocks)");
		for (const auto &f : other_)
			if (auto e = std::dynamic_pointer_cast<ElasticForm>(f))
			{
				if (elastic_)
					log_and_throw_error("NonsmoothNewton: selection = blind needs a single elastic form");
				elastic_ = e;
			}
		if (!elastic_ || !elastic_->supports_local())
			log_and_throw_error("NonsmoothNewton: selection = blind needs an elastic form with local element derivatives");
		const Vec x0 = full_size_ ? Vec(Vec::Zero(problem_.full_size())) : problem_.reduced_to_full(Vec::Zero(problem_.reduced_size()));
		// the inertia with a consistent mass couples the nodes of an element: its element blocks are terms of the element units
		for (const auto &f : other_)
			if (auto in = std::dynamic_pointer_cast<InertiaForm>(f); in && in->enabled() && !is_node_block_diagonal(in->mass(), 3))
				inertia_ = in;
		std::vector<std::shared_ptr<Form>> skip;
		if (inertia_)
		{
			if (!mass || mass->n != int(elastic_->n_elements()))
				log_and_throw_error("NonsmoothNewton: selection = blind with the consistent mass needs the element blocks of the mass matrix");
			if (const double e = mass->mismatch(inertia_->mass(), 3); !(e <= 1e-14))
				log_and_throw_error("NonsmoothNewton: the element blocks of the mass matrix do not add up to the mass matrix of the inertia form (relative difference {:g})", e);
			mass_ = mass;
			skip.push_back(inertia_);
		}
		node_local_ = BlindForms::classify(other_, elastic_, contact_, std::shared_ptr<Form>(), x0, 3, skip, session_->params().unsupported_remainder).node_local;
		if (full_size_)
		{
			// the penalty terms of the constraints are node terms if their Hessian couples only the dofs of one node
			StiffnessMatrix Hp(x0.size(), x0.size());
			problem_.add_penalty_hessian(x0, Hp);
			if (!is_node_block_diagonal(Hp, 3) && !session_->params().unsupported_remainder)
				log_and_throw_error("NonsmoothNewton: selection = blind cannot split the augmented Lagrangian penalty of the constraints into units (a constraint couples several nodes); blind/unsupported_forms = remainder keeps it in the frozen part");
		}
		UnitCache &cache = session_->cache();
		cache.set_mass(mass_);
		cache.bind(elastic_, 3, x0);
		if (!cache.on())
			log_and_throw_error("NonsmoothNewton: selection = blind needs P1 tetrahedra with unit dof weights and an elastic form with local derivatives");
		for (int e = 0; e < cache.n_elements(); ++e)
			element_of_key_.emplace(cache.element_key(e), e);
		selector_ = std::make_unique<UnitSelector>(problem_, *session_, *this);
		first_layer_blind_ = std::make_unique<FirstLayer>(problem_, fast_validity_, static_cast<::polyfem::solver::UnitSink &>(sink_));
	}

	BlindUnitTerm::~BlindUnitTerm() = default;

	// -------------------------------------------------------------------------------------------------------------- keys
	UnitKey BlindUnitTerm::collision_key(const ipc::NormalCollisions &cs, const size_t i) const
	{
		const auto vis = cs[i].vertex_ids(mesh_.edges(), mesh_.faces());
		const int nv = cs[i].num_vertices();
		int nodes[4];
		for (int a = 0; a < nv; ++a)
			nodes[a] = vis[a] >= 0 ? node_of_vertex_[vis[a]] : -1;
		return make_unit_key(nodes, nv);
	}

	void BlindUnitTerm::blind_exact_list(const ipc::NormalCollisions &cs, std::vector<int> &idx) const
	{
		idx.clear();
		const auto &A = session_->cache().cur();
		const bool have = A.u_ok;
		idx.reserve(cs.size());
		for (size_t i = 0; i < cs.size(); ++i)
		{
			const UnitKey k = collision_key(cs, i);
			const bool in_set = session_->in_S(k);
			// new at this point: always evaluated exactly (the frozen model has no information on it)
			const bool born = have && !in_set && A.coll.find(k) < 0;
			if (!have || in_set || born)
				idx.push_back(int(i));
		}
	}

	const std::vector<int> &BlindUnitTerm::exact_elements()
	{
		if (sel_ver_ != session_->version())
		{
			sel_ids_.clear();
			for (const auto &kv : session_->units())
			{
				const auto it = element_of_key_.find(kv.first);
				if (it != element_of_key_.end())
					sel_ids_.push_back(it->second);
				else if (kv.first.size() == 1)
					throw std::runtime_error("NonsmoothNewton: a node unit became exact (the node terms are quadratic: their force error is zero)");
			}
			std::sort(sel_ids_.begin(), sel_ids_.end());
			sel_ver_ = session_->version();
			rebuild_jratio();
		}
		return sel_ids_;
	}

	void BlindUnitTerm::rebuild_jratio()
	{
		// the Jacobian bound sees the elements that are not exact (the floor and the validity stay over all); Jg: all if set to 1
		const bool all = jratio_dyn_ == 1;
		if (fast_validity_ && !sel_ids_.empty() && !all)
		{
			jratio_mask_.assign(fast_validity_->n_tets(), 1);
			for (const int e : sel_ids_)
				if (e < int(jratio_mask_.size()))
					jratio_mask_[e] = 0;
		}
		else
			jratio_mask_.clear();
	}

	// ---------------------------------------------------------------------------------------------------- the units' data
	void BlindUnitTerm::fill_collisions(CollData &cd, const Vec &xfull)
	{
		cd.clear();
		const auto V = contact_->compute_displaced_surface(xfull);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight() / contact_->scale();
		const size_t nc = cs.size();
		cd.key.resize(nc);
		cd.nd.resize(nc);
		cd.dof.assign(nc * NL, -1);
		cd.g.assign(nc * NL, 0.0);
		cd.H.assign(nc * NL * NL, 0.0);
		cd.e.assign(nc, 0.0);
		utils::maybe_parallel_for(int(nc), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				const auto &col = cs[i];
				const ipc::VectorMax12d xc = col.dof(V, E, Fc);
				const int nd = int(xc.size());
				const auto vis = col.vertex_ids(E, Fc);
				const int nv = col.num_vertices();
				cd.key[i] = collision_key(cs, i);
				cd.nd[i] = (signed char)nd;
				const ipc::VectorMax12d g = w * bp.gradient(col, xc);
				for (int j = 0; j < nd; ++j)
					cd.g[i * NL + j] = g[j];
				Eigen::Map<Eigen::MatrixXd>(cd.H.data() + i * NL * NL, nd, nd) = w * bp.hessian(col, xc, ipc::PSDProjectionMethod::NONE);
				cd.e[i] = w * bp(col, xc);
				for (int a = 0; a < nv; ++a)
					for (int d = 0; d < 3; ++d)
						cd.dof[i * NL + 3 * a + d] = vis[a] >= 0 ? 3 * node_of_vertex_[vis[a]] + d : -1;
			}
		});
		cd.index.reserve(nc * 2);
		for (size_t i = 0; i < nc; ++i)
			cd.index.emplace(cd.key[i], int(i)); // the first pair of a node set represents it
		if (friction_)
			add_friction_terms(cd, xfull);
	}

	UnitKey BlindUnitTerm::tangent_key(const ipc::TangentialCollisions &fcs, const size_t i) const
	{
		const auto vis = fcs[i].vertex_ids(mesh_.edges(), mesh_.faces());
		const int nv = fcs[i].num_vertices();
		int nodes[4];
		for (int a = 0; a < nv; ++a)
			nodes[a] = vis[a] >= 0 ? node_of_vertex_[vis[a]] : -1;
		return make_unit_key(nodes, nv);
	}

	/// The friction terms of the lagged pairs at xfull join the entries of the same nodes (the same stencil as the collision pair they come from,
	/// so the same local dofs); a pair without a collision pair is an entry of its own.
	void BlindUnitTerm::add_friction_terms(CollData &cd, const Vec &xfull)
	{
		const auto &fcs = friction_->friction_collision_set();
		const size_t nf = fcs.size();
		cd.Hf.assign(cd.key.size() * NL * NL, 0.0);
		if (nf == 0)
			return;
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const Eigen::MatrixXd Vv = friction_->compute_surface_velocities(xfull);
		const auto &fp = friction_->friction_potential();
		const double cf = friction_->derivative_factor(), dvdx = friction_->dv_dx();
		struct Term
		{
			UnitKey key;
			int nd = 0;
			std::array<int, NL> dof;
			double g[NL];
			Eigen::MatrixXd H;
			double e = 0;
		};
		std::vector<Term> terms(nf);
		utils::maybe_parallel_for(int(nf), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				const auto &col = fcs[i];
				const ipc::VectorMax12d vc = col.dof(Vv, E, Fc);
				Term &t = terms[i];
				t.nd = int(vc.size());
				t.key = tangent_key(fcs, i);
				const auto vis = col.vertex_ids(E, Fc);
				t.dof.fill(-1);
				for (int a = 0; a < col.num_vertices(); ++a)
					for (int d = 0; d < 3; ++d)
						t.dof[3 * a + d] = vis[a] >= 0 ? 3 * node_of_vertex_[vis[a]] + d : -1;
				const ipc::VectorMax12d g = cf * fp.gradient(col, vc);
				for (int j = 0; j < t.nd; ++j)
					t.g[j] = g[j];
				t.H = (cf * dvdx) * fp.hessian(col, vc, ipc::PSDProjectionMethod::NONE);
				t.e = cf * fp(col, vc) / dvdx;
			}
		});
		const size_t nb = cd.key.size();
		for (size_t i = 0; i < nf; ++i)
		{
			const Term &t = terms[i];
			int entry = -1;
			if (const int j = cd.find(t.key); j >= 0 && j < int(nb) && cd.nd[j] == t.nd && std::equal(t.dof.begin(), t.dof.begin() + t.nd, cd.dof.begin() + std::size_t(j) * NL))
				entry = j;
			if (entry < 0)
			{
				entry = int(cd.key.size());
				cd.key.push_back(t.key);
				cd.nd.push_back((signed char)t.nd);
				cd.dof.resize(cd.dof.size() + NL, -1);
				std::copy(t.dof.begin(), t.dof.end(), cd.dof.begin() + std::size_t(entry) * NL);
				cd.g.resize(cd.g.size() + NL, 0.0);
				cd.H.resize(cd.H.size() + NL * NL, 0.0);
				cd.Hf.resize(cd.Hf.size() + NL * NL, 0.0);
				cd.e.push_back(0.0);
				cd.index.emplace(t.key, entry);
			}
			for (int j = 0; j < t.nd; ++j)
				cd.g[std::size_t(entry) * NL + j] += t.g[j];
			Eigen::Map<Eigen::MatrixXd>(cd.H.data() + std::size_t(entry) * NL * NL, t.nd, t.nd) += t.H;
			Eigen::Map<Eigen::MatrixXd>(cd.Hf.data() + std::size_t(entry) * NL * NL, t.nd, t.nd) += t.H;
			cd.e[entry] += t.e;
		}
	}

	UnitCache::Anchor &BlindUnitTerm::blind_capture(const Vec &xfull)
	{
		UnitCache &cache = session_->cache();
		bool is_new = false;
		UnitCache::Anchor &A = cache.anchor_for(xfull, elastic_->weight() / elastic_->scale(), &is_new);
		if (is_new)
			session_->clear_promoted(); // units made exact for another anchor are not part of the update of this one
		if (!A.u_ok || A.gen != session_->generation())
		{
			fill_collisions(A.coll, xfull);
			fill_node_data(node_local_, xfull, 3, A.node);
			if (full_size_)
				add_penalty_to_nodes(xfull, A.node);
			if (inertia_)
				fill_inertia(xfull, A);
			A.u_ok = true;
			A.gen = session_->generation();
		}
		return A;
	}

	void BlindUnitTerm::add_penalty_to_nodes(const Vec &xf, NodeData &node) const
	{
		Vec g = Vec::Zero(xf.size());
		problem_.add_penalty_gradient(xf, g);
		StiffnessMatrix H(xf.size(), xf.size());
		problem_.add_penalty_hessian(xf, H);
		for (Eigen::Index r = 0; r < g.size(); ++r)
			node.g[std::size_t(r / 3) * 3 + r % 3] += g[r];
		for (int c = 0; c < H.outerSize(); ++c)
			for (StiffnessMatrix::InnerIterator it(H, c); it; ++it)
				if (it.row() / 3 == it.col() / 3)
					node.H[std::size_t(it.col() / 3) * 9 + (it.row() % 3) + 3 * (it.col() % 3)] += it.value();
	}

	/// PSD projection of the Hessian block of entry j: the barrier part is projected, the friction part is not
	Eigen::MatrixXd BlindUnitTerm::project_barrier_part(const CollData &cd, const int j, const Eigen::MatrixXd &H)
	{
		if (cd.Hf.empty())
			return ipc::project_to_psd(H);
		const int nd = int(H.rows());
		const Eigen::Map<const Eigen::MatrixXd> Hf(cd.Hf.data() + std::size_t(j) * NL * NL, nd, nd);
		return Eigen::MatrixXd(ipc::project_to_psd(Eigen::MatrixXd(H - Hf)) + Hf);
	}

	void BlindUnitTerm::friction_exact_list(std::vector<int> &idx) const
	{
		idx.clear();
		if (!friction_)
			return;
		const auto &fcs = friction_->friction_collision_set();
		for (size_t i = 0; i < fcs.size(); ++i)
			if (session_->in_S(tangent_key(fcs, i)))
				idx.push_back(int(i));
	}

	void BlindUnitTerm::fill_inertia(const Vec &xf, UnitCache::Anchor &A) const
	{
		const double w = inertia_->weight() / inertia_->scale();
		const Vec xt = inertia_->x_tilde();
		A.wm = w;
		A.ig.assign(std::size_t(mass_->n) * NL, 0.0);
		utils::maybe_parallel_for(mass_->n, [&](int start, int end, int /*thread_id*/) {
			for (int e = start; e < end; ++e)
				mass_->gradient(e, w, xf, xt, A.ig.data() + std::size_t(e) * NL);
		});
	}

	// ---------------------------------------------------------------------------------------------------- smooth part
	void BlindUnitTerm::smooth_gradient(const Vec &x, Vec &grad)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		UnitCache &cache = session_->cache();
		blind_capture(xfull); // the units of the anchor (collisions, node terms) next to the elements' unit cache
		Vec gn = Vec::Zero(xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			Vec gt;
			if (f.get() == static_cast<Form *>(elastic_.get()))
			{
				{
					CaptureScope scope(cache.capture_begin(xfull, elastic_->weight() / elastic_->scale(), true));
					f->first_derivative(xfull, gt);
				}
				cache.capture_end(true);
			}
			else
				f->first_derivative(xfull, gt);
			gn += gt;
		}
		const std::vector<int> &ids = exact_elements();
		if (!ids.empty())
		{
			// the exact elements leave the smooth part
			Vec gs;
			if (cache.cur_has(xfull, true))
			{
				gs = Vec::Zero(xfull.size());
				const double f = cache.cur_f();
				const auto &A = cache.cur();
				for (const int e : ids)
				{
					const double *g = A.g.data() + std::size_t(e) * NL;
					const int *d = cache.dofs(e);
					for (int j = 0; j < NL; ++j)
						gs[d[j]] += (g[j] * f) * 1.0;
				}
			}
			else
				elastic_->subset_gradient(xfull, ids, gs);
			gn -= gs;
			if (inertia_)
			{
				// and their inertia terms
				const auto &A = cache.cur();
				for (const int e : ids)
					for (int j = 0; j < NL; ++j)
						gn[mass_->dof(e, j / 3, j % 3)] -= A.ig[std::size_t(e) * NL + j];
			}
		}
		{
			// the collision pairs of the anchor that are not exact: their gradient at the anchor
			const CollData &cd = cache.cur().coll;
			for (int j = 0; j < cd.size(); ++j)
			{
				if (session_->in_S(cd.key[j]))
					continue;
				for (int a = 0; a < cd.nd[j]; ++a)
					if (const int d = cd.dof[std::size_t(j) * NL + a]; d >= 0)
						gn[d] += cd.g[std::size_t(j) * NL + a];
			}
		}
		grad = to_reduced(gn);
		problem_.add_penalty_gradient(x, grad);
	}

	void BlindUnitTerm::smooth_hessian(const Vec &x, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		UnitCache &cache = session_->cache();
		blind_capture(xfull);
		for (auto &f : other_)
			f->set_project_to_psd(project_to_psd);
		StiffnessMatrix h(xfull.size(), xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			StiffnessMatrix tmp;
			if (f.get() == static_cast<Form *>(elastic_.get()))
			{
				{
					CaptureScope scope(cache.capture_begin(xfull, elastic_->weight() / elastic_->scale(), false));
					f->second_derivative(xfull, tmp);
				}
				cache.capture_end(false);
			}
			else
				f->second_derivative(xfull, tmp);
			h += tmp;
		}
		const std::vector<int> &ids = exact_elements();
		if (!ids.empty())
		{
			StiffnessMatrix hs;
			if (cache.cur_has(xfull, false))
			{
				const double f = cache.cur_f();
				const auto &A = cache.cur();
				std::vector<Eigen::Triplet<double>> trip;
				trip.reserve(ids.size() * 144);
				for (const int e : ids)
				{
					Eigen::Matrix<double, NL, NL> H = Eigen::Map<const Eigen::Matrix<double, NL, NL>>(A.H.data() + std::size_t(e) * NL * NL) * f;
					if (project_to_psd)
						H = ipc::project_to_psd(Eigen::MatrixXd(H));
					const int *d = cache.dofs(e);
					for (int i = 0; i < NL; ++i)
						for (int j = 0; j < NL; ++j)
							trip.emplace_back(d[i], d[j], H(i, j) * 1.0 * 1.0);
				}
				hs.resize(xfull.size(), xfull.size());
				hs.setFromTriplets(trip.begin(), trip.end());
			}
			else
				elastic_->subset_hessian(xfull, ids, project_to_psd, hs);
			h -= hs;
			if (inertia_)
			{
				std::vector<Eigen::Triplet<double>> trip;
				trip.reserve(ids.size() * 144);
				const double wm = inertia_->weight() / inertia_->scale();
				for (const int e : ids)
				{
					double Hm[144] = {0};
					mass_->add_hessian(e, wm, Hm);
					for (int i = 0; i < NL; ++i)
						for (int j = 0; j < NL; ++j)
							if (Hm[i + NL * j] != 0)
								trip.emplace_back(mass_->dof(e, i / 3, i % 3), mass_->dof(e, j / 3, j % 3), Hm[i + NL * j]);
				}
				StiffnessMatrix hm(xfull.size(), xfull.size());
				hm.setFromTriplets(trip.begin(), trip.end());
				h -= hm;
			}
		}
		{
			const CollData &cd = cache.cur().coll;
			std::vector<Eigen::Triplet<double>> trip;
			trip.reserve(std::size_t(cd.size()) * 100);
			for (int j = 0; j < cd.size(); ++j)
			{
				if (session_->in_S(cd.key[j]))
					continue;
				const int nd = cd.nd[j];
				Eigen::MatrixXd Hc = Eigen::Map<const Eigen::MatrixXd>(cd.H.data() + std::size_t(j) * NL * NL, nd, nd);
				if (project_to_psd)
					Hc = project_barrier_part(cd, j, Hc);
				for (int a = 0; a < nd; ++a)
				{
					const int da = cd.dof[std::size_t(j) * NL + a];
					if (da < 0)
						continue;
					for (int b = 0; b < nd; ++b)
						if (const int db = cd.dof[std::size_t(j) * NL + b]; db >= 0)
							trip.emplace_back(da, db, Hc(a, b));
				}
			}
			StiffnessMatrix hc(xfull.size(), xfull.size());
			hc.setFromTriplets(trip.begin(), trip.end());
			h += hc;
		}
		to_reduced(h);
		problem_.add_penalty_hessian(x, h);
		hessian = std::move(h);
	}

	// ---------------------------------------------------------------------------------------------------- exact term
	double BlindUnitTerm::value(const Vec &y)
	{
		const Vec yf = problem_.reduced_to_full(y);
		double elastic = 0;
		const std::vector<int> &ids = exact_elements();
		if (!ids.empty())
		{
			elastic = elastic_->subset_energy(yf, ids);
			if (inertia_)
			{
				const double wm = inertia_->weight() / inertia_->scale();
				const Vec xt = inertia_->x_tilde();
				for (const int e : ids)
					elastic += mass_->energy(e, wm, yf, xt);
			}
		}
		const auto V = contact_->compute_displaced_surface(yf);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight() / contact_->scale();
		std::vector<int> idx;
		blind_exact_list(cs, idx);
		double s = 0;
		for (const int i : idx)
			s += bp(cs[i], cs[i].dof(V, E, Fc));
		double friction = 0;
		if (friction_)
		{
			// the lagged friction pairs of the exact units, at the velocities of yf
			std::vector<int> fidx;
			friction_exact_list(fidx);
			const auto &fcs = friction_->friction_collision_set();
			const Eigen::MatrixXd Vv = friction_->compute_surface_velocities(yf);
			const auto &fp = friction_->friction_potential();
			double sf = 0;
			for (const int i : fidx)
				sf += fp(fcs[i], fcs[i].dof(Vv, E, Fc));
			friction = friction_->derivative_factor() * sf / friction_->dv_dx();
		}
		return w * s + elastic + friction;
	}

	void BlindUnitTerm::gradient(const Vec &y, Vec &grad)
	{
		const Vec yf = problem_.reduced_to_full(y);
		Vec g = Vec::Zero(yf.size());
		const auto V = contact_->compute_displaced_surface(yf);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight() / contact_->scale();
		std::vector<int> idx;
		blind_exact_list(cs, idx);
		for (const int i : idx)
		{
			const auto &col = cs[i];
			const ipc::VectorMax12d xc = col.dof(V, E, Fc);
			const ipc::VectorMax12d gc = w * bp.gradient(col, xc);
			const auto vis = col.vertex_ids(E, Fc);
			for (int a = 0; a < col.num_vertices(); ++a)
				for (int d = 0; d < 3; ++d)
					if (vis[a] >= 0)
						g[3 * node_of_vertex_[vis[a]] + d] += gc[3 * a + d];
		}
		if (friction_)
		{
			std::vector<int> fidx;
			friction_exact_list(fidx);
			const auto &fcs = friction_->friction_collision_set();
			const Eigen::MatrixXd Vv = friction_->compute_surface_velocities(yf);
			const auto &fp = friction_->friction_potential();
			const double cf = friction_->derivative_factor();
			for (const int i : fidx)
			{
				const auto &col = fcs[i];
				const ipc::VectorMax12d gc = cf * fp.gradient(col, col.dof(Vv, E, Fc));
				const auto vis = col.vertex_ids(E, Fc);
				for (int a = 0; a < col.num_vertices(); ++a)
					for (int d = 0; d < 3; ++d)
						if (vis[a] >= 0)
							g[3 * node_of_vertex_[vis[a]] + d] += gc[3 * a + d];
			}
		}
		const std::vector<int> &ids = exact_elements();
		if (!ids.empty())
		{
			Vec ge;
			elastic_->subset_gradient(yf, ids, ge);
			g += ge;
			if (inertia_)
			{
				const double wm = inertia_->weight() / inertia_->scale();
				const Vec xt = inertia_->x_tilde();
				double gl[NL];
				for (const int e : ids)
				{
					mass_->gradient(e, wm, yf, xt, gl);
					for (int j = 0; j < NL; ++j)
						g[mass_->dof(e, j / 3, j % 3)] += gl[j];
				}
			}
		}
		grad = to_reduced(g);
	}

	void BlindUnitTerm::hessian(const Vec &y, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		StiffnessMatrix K;
		const Vec yf = problem_.reduced_to_full(y);
		const auto V = contact_->compute_displaced_surface(yf);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight() / contact_->scale();
		std::vector<int> idx;
		blind_exact_list(cs, idx);
		std::vector<Eigen::Triplet<double>> trip;
		trip.reserve(idx.size() * 100);
		for (const int i : idx)
		{
			const auto &col = cs[i];
			const ipc::VectorMax12d xc = col.dof(V, E, Fc);
			Eigen::MatrixXd Hc = w * bp.hessian(col, xc, ipc::PSDProjectionMethod::NONE);
			if (project_to_psd)
				Hc = ipc::project_to_psd(Hc);
			const auto vis = col.vertex_ids(E, Fc);
			const int nv = col.num_vertices();
			for (int a = 0; a < nv; ++a)
				for (int da = 0; da < 3; ++da)
					for (int b = 0; b < nv; ++b)
						for (int db = 0; db < 3; ++db)
							if (vis[a] >= 0 && vis[b] >= 0)
								trip.emplace_back(3 * node_of_vertex_[vis[a]] + da, 3 * node_of_vertex_[vis[b]] + db, Hc(3 * a + da, 3 * b + db));
		}
		// the friction of the exact units: not projected; B of the first solve of a time step takes it at the velocities of the previous step
		const bool at_lag = friction_ && project_to_psd && problem_.take_friction_b_lag();
		if (friction_)
		{
			std::vector<int> fidx;
			friction_exact_list(fidx);
			const auto &fcs = friction_->friction_collision_set();
			const Eigen::MatrixXd Vv = at_lag ? friction_->previous_surface_velocities() : friction_->compute_surface_velocities(yf);
			const auto &fp = friction_->friction_potential();
			const double cfd = friction_->derivative_factor() * friction_->dv_dx();
			for (const int i : fidx)
			{
				const auto &col = fcs[i];
				const Eigen::MatrixXd Hc = cfd * fp.hessian(col, col.dof(Vv, E, Fc), ipc::PSDProjectionMethod::NONE);
				const auto vis = col.vertex_ids(E, Fc);
				const int nv = col.num_vertices();
				for (int a = 0; a < nv; ++a)
					for (int da = 0; da < 3; ++da)
						for (int b = 0; b < nv; ++b)
							for (int db = 0; db < 3; ++db)
								if (vis[a] >= 0 && vis[b] >= 0)
									trip.emplace_back(3 * node_of_vertex_[vis[a]] + da, 3 * node_of_vertex_[vis[b]] + db, Hc(3 * a + da, 3 * b + db));
			}
		}
		K.resize(yf.size(), yf.size());
		K.setFromTriplets(trip.begin(), trip.end());
		const std::vector<int> &ids = exact_elements();
		if (!ids.empty())
		{
			StiffnessMatrix Ke;
			elastic_->subset_hessian(yf, ids, project_to_psd, Ke);
			K += Ke;
			if (inertia_)
			{
				// the inertia terms are not projected (positive semidefinite)
				std::vector<Eigen::Triplet<double>> trip;
				const double wm = inertia_->weight() / inertia_->scale();
				for (const int e : ids)
				{
					double Hm[144] = {0};
					mass_->add_hessian(e, wm, Hm);
					for (int i = 0; i < NL; ++i)
						for (int j = 0; j < NL; ++j)
							if (Hm[i + NL * j] != 0)
								trip.emplace_back(mass_->dof(e, i / 3, i % 3), mass_->dof(e, j / 3, j % 3), Hm[i + NL * j]);
				}
				StiffnessMatrix Km(yf.size(), yf.size());
				Km.setFromTriplets(trip.begin(), trip.end());
				K += Km;
			}
		}
		to_reduced(K);
		hessian = std::move(K);
	}

	bool BlindUnitTerm::hessian_blocks(const Vec &y, Vec &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks)
	{
		if (!(opts_.local_blocks && local_ok_) || problem_.uses_full_size() != full_size_)
			return false; // the local blocks map to the dofs of the size of the term
		const Vec yf = problem_.reduced_to_full(y);
		const auto V = contact_->compute_displaced_surface(yf);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight() / contact_->scale();
		grad = Vec::Zero(y.size());
		std::vector<int> bidx; // the collisions of the state that are exact
		blind_exact_list(cs, bidx);
		std::vector<int> fidx; // the lagged friction pairs of the exact units
		friction_exact_list(fidx);
		const std::vector<int> &ids = exact_elements();
		const size_t nb = bidx.size(), nf = fidx.size(), ne = ids.size();
		blocks.clear();
		blocks.resize(nb + nf + ne);
		std::vector<ipc::VectorMax12d> gb(nb + nf + ne);
		const auto set_indices = [&](const auto &vids, polysolve::nonlinear::HessianBlock &b) {
			for (int a = 0; a < b.size / 3; ++a)
				for (int d = 0; d < 3; ++d)
					b.indices[3 * a + d] = vids[a] >= 0 ? red_of_full_[3 * node_of_vertex_[vids[a]] + d] : -1;
		};
		utils::maybe_parallel_for(int(nb), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				const auto &col = cs[size_t(bidx[i])];
				const ipc::VectorMax12d xc = col.dof(V, E, Fc);
				polysolve::nonlinear::HessianBlock &b = blocks[i];
				b.size = int(xc.size());
				gb[i] = w * bp.gradient(col, xc);
				b.H.topLeftCorner(b.size, b.size) = w * bp.hessian(col, xc, ipc::PSDProjectionMethod::NONE);
				set_indices(col.vertex_ids(E, Fc), b);
			}
		});
		if (nf > 0)
		{
			// one block per friction pair, at the surface velocities of y (unprojected; gradient wrt x = weighted gradient wrt v, Hessian wrt x = dv/dx times
			// the weighted Hessian wrt v)
			const auto &fcs = friction_->friction_collision_set();
			const Eigen::MatrixXd Vv = friction_->compute_surface_velocities(yf);
			const auto &fp = friction_->friction_potential();
			const double cf = friction_->derivative_factor(), dvdx = friction_->dv_dx();
			utils::maybe_parallel_for(int(nf), [&](int start, int end, int /*thread_id*/) {
				for (size_t i = start; i < size_t(end); ++i)
				{
					const auto &col = fcs[size_t(fidx[i])];
					const ipc::VectorMax12d vc = col.dof(Vv, E, Fc);
					polysolve::nonlinear::HessianBlock &b = blocks[nb + i];
					b.size = int(vc.size());
					gb[nb + i] = cf * fp.gradient(col, vc);
					b.H.topLeftCorner(b.size, b.size) = (cf * dvdx) * fp.hessian(col, vc, ipc::PSDProjectionMethod::NONE);
					set_indices(col.vertex_ids(E, Fc), b);
				}
			});
		}
		if (ne > 0)
		{
			const Eigen::MatrixXd ym = yf;
			const double wm_blocks = inertia_ ? inertia_->weight() / inertia_->scale() : 0.0;
			const Vec xt_blocks = inertia_ ? inertia_->x_tilde() : Vec();
			struct TD
			{
				assembler::ElementAssemblyValues vals;
				ElasticForm::ElementLocal loc;
			};
			auto st = utils::create_thread_storage(TD());
			utils::maybe_parallel_for(int(ne), [&](int start, int end, int thread_id) {
				TD &td = utils::get_local_thread_storage(st, thread_id);
				for (int k = start; k < end; ++k)
				{
					elastic_->element_local(ym, ids[k], true, false, td.vals, td.loc);
					if (inertia_)
					{
						// the element's inertia terms join its block (the Hessian block is the raw sum; B projects the elastic part only)
						double gl[NL];
						mass_->gradient(ids[k], wm_blocks, yf, xt_blocks, gl);
						for (int j = 0; j < NL; ++j)
							td.loc.g[j] += gl[j];
						mass_->add_hessian(ids[k], wm_blocks, td.loc.H.data());
					}
					const int n = int(td.loc.g.size());
					if (n != NL)
						throw std::runtime_error("NonsmoothNewton: an exact element with more than 12 local dofs");
					polysolve::nonlinear::HessianBlock &b = blocks[nb + nf + k];
					b.size = n;
					gb[nb + nf + k] = td.loc.g;
					b.H.topLeftCorner(n, n) = td.loc.H;
					for (int j = 0; j < n; ++j)
					{
						if (td.loc.w[j] != 1.0)
							throw std::runtime_error("NonsmoothNewton: an exact element dof with weight != 1");
						b.indices[j] = td.loc.dofs[j] >= 0 ? red_of_full_[td.loc.dofs[j]] : -1;
					}
				}
			});
		}
		for (size_t i = 0; i < blocks.size(); ++i)
		{
			const auto &b = blocks[i];
			for (int j = 0; j < b.size; ++j)
				if (const int r = b.indices[j]; r >= 0)
					grad[r] += gb[i][j];
		}
		return true;
	}

	// ------------------------------------------------------------------------------------------------------ born units
	void BlindUnitTerm::step_accepted(const Vec &x0, const Vec &x1)
	{
		const Vec x0f = problem_.reduced_to_full(x0);
		const auto &A = session_->cache().cur();
		if (!(A.u_ok && A.xf.size() == x0f.size() && A.xf == x0f))
			return; // this step did not start at the anchor of a model
		// the collision pairs of the accepted point that are new (not at the anchor): they were exact on the path and stay exact
		const auto &cs = contact_->collision_set();
		std::vector<UnitKey> born;
		for (size_t i = 0; i < cs.size(); ++i)
		{
			const UnitKey k = collision_key(cs, i);
			if (A.coll.find(k) < 0 && !session_->in_S(k))
				born.push_back(k);
		}
		session_->add_keys(born, false);
		session_->mark_born(born);
	}

	// ------------------------------------------------------------------------------------------- previous anchor, again
	bool BlindUnitTerm::blind_reeval(const UnitCache::Anchor &prev, UnitCache::Anchor &out, const Vec &xcur)
	{
		if (!prev.g_ok || !prev.h_ok || !prev.u_ok)
			return false;
		const Vec &x0 = prev.xf;
		out = UnitCache::Anchor();
		out.xf = x0;
		out.f = 1.0; // weighted data
		out.projected = prev.projected;
		out.res2 = prev.res2;
		UnitCache &cache = session_->cache();
		{
			const int nel = cache.n_elements();
			out.g.assign(std::size_t(nel) * NL, 0.0);
			out.H.assign(std::size_t(nel) * NL * NL, 0.0);
			const Eigen::MatrixXd xm = x0;
			struct TD
			{
				assembler::ElementAssemblyValues vals;
				ElasticForm::ElementLocal loc;
			};
			auto st = utils::create_thread_storage(TD());
			utils::maybe_parallel_for(nel, [&](int start, int end, int thread_id) {
				TD &td = utils::get_local_thread_storage(st, thread_id);
				for (int e = start; e < end; ++e)
				{
					elastic_->element_local(xm, e, true, false, td.vals, td.loc);
					for (int j = 0; j < NL; ++j)
						out.g[std::size_t(e) * NL + j] = td.loc.g[j];
					for (int j = 0; j < NL * NL; ++j)
						out.H[std::size_t(e) * NL * NL + j] = td.loc.H.data()[j];
				}
			});
			out.g_ok = out.h_ok = true;
		}
		{
			// the collision pairs of the previous anchor with the barrier of now (the collision state of the problem goes back to xcur)
			CollData tmp;
			contact_->solution_changed(x0);
			fill_collisions(tmp, x0);
			contact_->solution_changed(xcur);
			out.coll = prev.coll; // keys, dofs, sizes, index of the previous anchor's list
			CollData &cd = out.coll;
			for (int j = 0; j < cd.size(); ++j)
			{
				const int i = tmp.find(cd.key[j]);
				if (i < 0 || tmp.nd[i] != cd.nd[j])
				{
					std::fill(cd.g.begin() + std::size_t(j) * NL, cd.g.begin() + std::size_t(j + 1) * NL, 0.0);
					std::fill(cd.H.begin() + std::size_t(j) * NL * NL, cd.H.begin() + std::size_t(j + 1) * NL * NL, 0.0);
					if (!cd.Hf.empty())
						std::fill(cd.Hf.begin() + std::size_t(j) * NL * NL, cd.Hf.begin() + std::size_t(j + 1) * NL * NL, 0.0);
					cd.e[j] = 0;
					continue;
				}
				std::copy(tmp.g.begin() + std::size_t(i) * NL, tmp.g.begin() + std::size_t(i + 1) * NL, cd.g.begin() + std::size_t(j) * NL);
				std::copy(tmp.H.begin() + std::size_t(i) * NL * NL, tmp.H.begin() + std::size_t(i + 1) * NL * NL, cd.H.begin() + std::size_t(j) * NL * NL);
				if (!cd.Hf.empty() && tmp.Hf.size() == tmp.H.size())
					std::copy(tmp.Hf.begin() + std::size_t(i) * NL * NL, tmp.Hf.begin() + std::size_t(i + 1) * NL * NL, cd.Hf.begin() + std::size_t(j) * NL * NL);
				cd.e[j] = tmp.e[i];
			}
		}
		fill_node_data(node_local_, x0, 3, out.node);
		if (full_size_)
			add_penalty_to_nodes(x0, out.node);
		if (inertia_)
			fill_inertia(x0, out);
		out.u_ok = true;
		out.gen = session_->generation();
		return true;
	}

	// ---------------------------------------------------------------------------------------------------- model change
	bool BlindUnitTerm::promotion_delta(const Vec &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta)
	{
		const std::vector<UnitKey> pkeys = session_->take_promoted();
		if (pkeys.empty())
			return false;
		UnitCache &cache = session_->cache();
		const Vec xf = problem_.reduced_to_full(x);
		const int n = int(x.size());
		const UnitCache::Anchor &A = cache.cur();
		if (!(A.u_ok && A.xf.size() == xf.size() && A.xf == xf))
			return false; // no unit data of this anchor
		std::vector<Eigen::Triplet<double>> trip;
		Vec dg = Vec::Zero(n);
		double dE = 0;
		int dn = 0;
		std::vector<int> ids, colls;
		for (const UnitKey &k : pkeys)
		{
			const auto it = element_of_key_.find(k);
			if (it != element_of_key_.end())
				ids.push_back(it->second);
			else if (k.size() == 1)
				throw std::runtime_error("NonsmoothNewton: a node unit became exact");
			else if (const int j = A.coll.find(k); j >= 0)
				colls.push_back(j); // a collision that is not at this anchor changes nothing in its model
		}
		std::sort(ids.begin(), ids.end());
		if (!ids.empty())
		{
			const bool cached = cache.cur_has(xf, true) && cache.cur_has(xf, false);
			const Eigen::MatrixXd xm = cached ? Eigen::MatrixXd() : Eigen::MatrixXd(xf);
			assembler::ElementAssemblyValues vals;
			ElasticForm::ElementLocal loc;
			const double f = cached ? cache.cur_f() : 1.0;
			for (const int e : ids)
			{
				Eigen::Matrix<double, NL, NL> H;
				Eigen::Matrix<double, NL, 1> g;
				std::array<int, NL> d;
				if (cached)
				{
					H = Eigen::Map<const Eigen::Matrix<double, NL, NL>>(A.H.data() + std::size_t(e) * NL * NL) * f;
					for (int j = 0; j < NL; ++j)
					{
						g[j] = A.g[std::size_t(e) * NL + j] * f;
						d[j] = cache.dofs(e)[j];
					}
				}
				else
				{
					elastic_->element_local(xm, e, true, false, vals, loc);
					H = loc.H;
					for (int j = 0; j < NL; ++j)
					{
						g[j] = loc.g[j];
						d[j] = loc.dofs[j];
					}
				}
				if (projected)
					H = ipc::project_to_psd(Eigen::MatrixXd(H));
				for (int a = 0; a < NL; ++a)
				{
					const int ra = d[a] >= 0 ? red_of_full_[d[a]] : -1;
					if (ra < 0)
						continue;
					dg[ra] += g[a];
					for (int b = 0; b < NL; ++b)
					{
						const int rb = d[b] >= 0 ? red_of_full_[d[b]] : -1;
						if (rb >= 0)
							trip.emplace_back(ra, rb, H(a, b));
					}
				}
			}
			dE += elastic_->subset_energy(xf, ids);
			dn += int(ids.size());
		}
		for (const int j : colls)
		{
			const CollData &cd = A.coll;
			const int nd = cd.nd[j];
			Eigen::MatrixXd Hc = Eigen::Map<const Eigen::MatrixXd>(cd.H.data() + std::size_t(j) * NL * NL, nd, nd);
			if (projected)
				Hc = project_barrier_part(cd, j, Hc);
			for (int a = 0; a < nd; ++a)
			{
				const int da = cd.dof[std::size_t(j) * NL + a];
				const int ra = da >= 0 ? red_of_full_[da] : -1;
				if (ra < 0)
					continue;
				dg[ra] += cd.g[std::size_t(j) * NL + a];
				for (int b = 0; b < nd; ++b)
				{
					const int db = cd.dof[std::size_t(j) * NL + b];
					const int rb = db >= 0 ? red_of_full_[db] : -1;
					if (rb >= 0)
						trip.emplace_back(ra, rb, Hc(a, b));
				}
			}
			dE += cd.e[j];
			++dn;
		}
		delta.dH.resize(n, n);
		delta.dH.setFromTriplets(trip.begin(), trip.end());
		delta.dg = dg;
		delta.dE = dE;
		delta.n = dn;
		return true;
	}

	// -------------------------------------------------------------------------------------------- limiting collisions
	int BlindUnitTerm::ccd_limiting()
	{
		if (certifier_ == nullptr || certifier_->limit_family < 0 || certifier_->limit_family > 3)
			return 0;
		const auto &L = certifier_->limit_vis;
		int nodes[4], nv = 0;
		for (int a = 0; a < 4; ++a)
			if (L[a] >= 0)
				nodes[nv++] = node_of_vertex_[L[a]];
		const UnitKey k = make_unit_key(nodes, nv);
		const auto &A = session_->cache().cur();
		if (!A.u_ok || session_->in_S(k) || A.coll.find(k) < 0)
			return 0; // exact, born, or a pair that is not a collision of the anchor: the limit stands
		session_->add_keys({k}, true);
		return 1;
	}

	int BlindUnitTerm::ftb_limiting(const Vec &a, const Vec &b, const double tau)
	{
		const auto Va = contact_->compute_displaced_surface(problem_.reduced_to_full(a));
		const auto Vb = contact_->compute_displaced_surface(problem_.reduced_to_full(b));
		const auto &cs = contact_->collision_set();
		const auto &Em = mesh_.edges();
		const auto &Fm = mesh_.faces();
		double best = std::numeric_limits<double>::infinity();
		int bi = -1;
		for (size_t i = 0; i < cs.size(); ++i)
		{
			const auto &c = cs[i];
			const Vec xa = c.dof(Va, Em, Fm), xb = c.dof(Vb, Em, Fm);
			const double d2 = c.compute_distance(xa);
			if (!(d2 > 0))
				continue;
			const double dist = std::sqrt(d2);
			const double rate = c.compute_distance_gradient(xa).dot(xb - xa) / (2 * dist);
			if (rate < 0)
			{
				const double sc = (1 - tau) * dist / -rate;
				if (sc < best)
				{
					best = sc;
					bi = int(i);
				}
			}
		}
		if (bi < 0 || !(best < 1))
			return 0;
		const auto &A = session_->cache().cur();
		const UnitKey k = collision_key(cs, size_t(bi));
		if (!A.u_ok || session_->in_S(k) || A.coll.find(k) < 0)
			return 0; // already exact, or born: the limit stands
		session_->add_keys({k}, true);
		return 1;
	}

	// ----------------------------------------------------------------------------------------------- repair of B
	void BlindUnitTerm::blind_repair_units(const Vec &x, std::vector<RepairUnit> &units)
	{
		UnitCache &cache = session_->cache();
		const Vec xf = problem_.reduced_to_full(x);
		const UnitCache::Anchor &A = cache.cur();
		if (!(A.u_ok && A.xf.size() == xf.size() && A.xf == xf))
			return;
		for (int e = 0; e < cache.n_elements(); ++e)
		{
			RepairUnit r;
			r.nd = NL;
			const UnitKey &k = cache.element_key(e);
			r.n_nodes = k.size();
			std::copy(k.node.begin(), k.node.end(), r.node.begin());
			for (int j = 0; j < NL; ++j)
				r.dof[j] = red_of_full_[cache.dofs(e)[j]];
			r.H = Eigen::Map<const Eigen::Matrix<double, NL, NL>>(A.H.data() + std::size_t(e) * NL * NL) * A.f;
			r.frozen = !session_->in_S(k);
			units.push_back(std::move(r));
		}
		const CollData &cd = A.coll;
		for (int j = 0; j < cd.size(); ++j)
		{
			RepairUnit r;
			r.nd = cd.nd[j];
			const UnitKey &k = cd.key[j];
			r.n_nodes = k.size();
			std::copy(k.node.begin(), k.node.end(), r.node.begin());
			for (int a = 0; a < r.nd; ++a)
				r.dof[a] = cd.dof[std::size_t(j) * NL + a] >= 0 ? red_of_full_[cd.dof[std::size_t(j) * NL + a]] : -1;
			r.H = Eigen::Map<const Eigen::MatrixXd>(cd.H.data() + std::size_t(j) * NL * NL, r.nd, r.nd);
			if (!cd.Hf.empty())
				r.H -= Eigen::Map<const Eigen::MatrixXd>(cd.Hf.data() + std::size_t(j) * NL * NL, r.nd, r.nd); // the friction part is not projected
			r.frozen = !session_->in_S(k);
			units.push_back(std::move(r));
		}
	}

	polysolve::nonlinear::BasisPort *BlindUnitTerm::basis()
	{
		if (!repair_port_)
			repair_port_ = std::make_unique<BasisRepair>(
				[this](const Vec &x, std::vector<RepairUnit> &units) { blind_repair_units(x, units); },
				[this](const long index) { return node_of_unknown(index); });
		return repair_port_.get();
	}

	// ------------------------------------------------------------------------------------------------------- the ports
	bool BlindUnitTerm::Sink::is_exact_element(const int e) const
	{
		return t_.session_->in_S(t_.session_->cache().element_key(e));
	}

	int BlindUnitTerm::Sink::add_elements(const std::vector<int> &ids)
	{
		std::vector<UnitKey> keys;
		keys.reserve(ids.size());
		for (const int e : ids)
			keys.push_back(t_.session_->cache().element_key(e));
		return t_.session_->add_keys(keys, true);
	}

	int BlindUnitTerm::Sink::add_limiting_contacts(const polysolve::nonlinear::LimitEvent &event)
	{
		if (event.kind == polysolve::nonlinear::LimitEvent::Kind::ccd)
			return t_.ccd_limiting();
		if (event.a == nullptr || event.b == nullptr)
			return 0;
		return t_.ftb_limiting(*event.a, *event.b, event.ftb_tau);
	}

	bool BlindUnitTerm::Sink::take_delta(const Vec &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta)
	{
		return t_.promotion_delta(x, projected, delta);
	}

	int BlindUnitTerm::Port::select(const Context &context, polysolve::nonlinear::NonsmoothDelta &delta)
	{
		if (t_.session_->wall_now())
		{
			// the wall switch: no second or third layer in this time step (the first layer and the born units go on)
			return 0;
		}
		const int n = t_.selector_->select(context);
		if (n > 0)
			t_.promotion_delta(context.x, context.projected, delta);
		return n;
	}

	bool BlindUnitTerm::Port::reread_allowed() const
	{
		return !(t_.session_->wall_now() && t_.session_->params().wall_reread_off);
	}

	void BlindUnitTerm::Port::jratio_exact(const int on)
	{
		const int v = on > 0 ? 1 : 0;
		if (t_.jratio_dyn_ == v)
			return;
		t_.jratio_dyn_ = v;
		t_.exact_elements();
		t_.rebuild_jratio();
	}

	double BlindUnitTerm::Port::min_ratio_exact(const Vec &x, const Vec &y)
	{
		const std::vector<int> &ids = t_.exact_elements();
		if (!t_.fast_validity_ || ids.empty())
			return NaN;
		const Vec xf = t_.problem_.reduced_to_full(x);
		if (!(t_.jdet_key_.size() == xf.size() && t_.jdet_key_ == xf))
		{
			t_.jdet_ = t_.fast_validity_->dets(xf);
			t_.jdet_key_ = xf;
		}
		std::vector<char> mask(t_.fast_validity_->n_tets(), 0);
		for (const int e : ids)
			mask[e] = 1;
		double mj = NaN, mr = NaN;
		t_.fast_validity_->min_jacobians(t_.jdet_, t_.problem_.reduced_to_full(y), false, true, mj, mr, &mask);
		return mr;
	}

	void BlindUnitTerm::Port::outer_end(const long *cuts, const long inner)
	{
		t_.session_->on_outer_end(cuts, inner);
	}
} // namespace polyfem::solver
