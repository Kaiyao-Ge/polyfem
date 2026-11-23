#include "BarrierContactNonsmoothTerm.hpp"

#include "BilateralSegmentCertifier.hpp"
#include "FastTetValidity.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/nonsmooth/BasisRepair.hpp>
#include <polyfem/solver/forms/Form.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <ipc/candidates/candidates.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/potentials/barrier_potential.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace polyfem::solver
{
	namespace
	{
		using Vec = Eigen::VectorXd;
		constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

		/// null control on the candidate order: 1 = reverse, s >= 2 = shuffle seeded by s
		void candidate_permute(ipc::Candidates &C, int s)
		{
			if (s == 1)
			{
				std::reverse(C.ee_candidates.begin(), C.ee_candidates.end());
				std::reverse(C.fv_candidates.begin(), C.fv_candidates.end());
				return;
			}
			std::mt19937_64 g{static_cast<uint64_t>(s)};
			std::shuffle(C.ee_candidates.begin(), C.ee_candidates.end(), g);
			std::shuffle(C.fv_candidates.begin(), C.fv_candidates.end(), g);
		}

		template <typename T>
		T value_or(const json &j, const char *key, const T &def)
		{
			return j.is_object() && j.contains(key) ? j[key].get<T>() : def;
		}
	} // namespace

	bool selects_nonsmooth_newton(const json &nonlinear_params)
	{
		if (!nonlinear_params.is_object() || !nonlinear_params.contains("solver"))
			return false;
		const json &s = nonlinear_params["solver"];
		if (s.is_string())
			return s.get<std::string>() == "NonsmoothNewton";
		if (s.is_array())
			for (const auto &e : s)
				if (e.is_object() && e.contains("type") && e["type"] == "NonsmoothNewton")
					return true;
		return false;
	}

	BarrierContactNonsmoothTerm::Options BarrierContactNonsmoothTerm::Options::from_json(const json &nonlinear_params)
	{
		Options o;
		if (!nonlinear_params.is_object() || !nonlinear_params.contains("NonsmoothNewton"))
			return o;
		const json &p = nonlinear_params["NonsmoothNewton"];
		if (p.is_object() && p.contains("first_layer"))
		{
			const json &f = p["first_layer"];
			const std::string enabled = value_or(f, "enabled", std::string("auto"));
			o.first_layer = enabled == "on" || (enabled == "auto" && value_or(p, "selection", std::string("energy_split")) == "blind");
			o.lift_model = value_or(f, "lift_model", o.lift_model);
			o.lift_basis = value_or(f, "lift_basis", o.lift_basis);
		}
		if (!p.is_object() || !p.contains("debug"))
			return o;
		const json &d = p["debug"];
		o.certificate = value_or(d, "certificate", o.certificate);
		o.dedupe = value_or(d, "dedupe", o.dedupe);
		o.local_blocks = value_or(d, "local_blocks", o.local_blocks);
		o.fast_validity = value_or(d, "fast_validity", o.fast_validity);
		o.candidate_permutation = value_or(d, "candidate_permutation", o.candidate_permutation);
		return o;
	}

	struct BarrierContactNonsmoothTerm::AnchorHolder
	{
		FastTetValidity::Anchor anchor;
	};

	BarrierContactNonsmoothTerm::BarrierContactNonsmoothTerm(NLProblem &problem,
															 const std::vector<std::shared_ptr<Form>> &forms,
															 const Eigen::VectorXd &mass_full,
															 const ipc::CollisionMesh &collision_mesh,
															 const Eigen::SparseMatrix<double> *displacement_map,
															 const std::shared_ptr<const FastTetValidity> &fast_validity,
															 const Options &opts,
															 const std::shared_ptr<LiftSet> &lift)
		: lift_(lift), problem_(problem), mesh_(collision_mesh), displacement_map_(displacement_map), fast_validity_(fast_validity),
		  opts_(opts), mass_full_(mass_full), anchor_(std::make_unique<AnchorHolder>())
	{
		// the step bound of the certified narrow phase combines all other forms (friction included), as without the term
		std::vector<std::shared_ptr<Form>> not_contact;
		for (const auto &f : forms)
		{
			if (auto c = std::dynamic_pointer_cast<BarrierContactForm>(f))
			{
				if (contact_)
					log_and_throw_error("NonsmoothNewton: more than one barrier contact form");
				contact_ = c;
				continue;
			}
			not_contact.push_back(f);
			if (auto fr = std::dynamic_pointer_cast<FrictionForm>(f))
			{
				if (friction_)
					log_and_throw_error("NonsmoothNewton: more than one friction form");
				friction_ = fr;
				continue;
			}
			other_.push_back(f);
		}
		if (!contact_)
			log_and_throw_error("NonsmoothNewton: the problem has no barrier contact form");
		certifier_ = std::make_unique<BilateralSegmentCertifier>(problem_, contact_, not_contact, mesh_, opts_.dedupe);
		const auto &Vr = mesh_.rest_positions();
		const auto &Ed = mesh_.edges();
		double sum = 0;
		for (int e = 0; e < Ed.rows(); ++e)
			sum += (Vr.row(Ed(e, 1)) - Vr.row(Ed(e, 0))).norm();
		h_edge_ = Ed.rows() ? sum / Ed.rows() : NaN;
		if (opts_.local_blocks)
			init_local_map();
		if (opts_.first_layer && lift_ != nullptr)
			init_lift();
	}

	BarrierContactNonsmoothTerm::~BarrierContactNonsmoothTerm() = default;

	// ------------------------------------------------------------------------------------------------------- first layer
	// The elastic elements that caused a cut of the path (a Jacobian limit) are the lifted elements: they leave the smooth part and
	// the exact term gets their energy, gradient and Hessian blocks (lift_model exact), or they stay in the smooth part and only B
	// (lift_basis projected: its correction dB, lift_model frozen) or the model's quadratic (frozen_projected) takes their
	// projected block. The lift set lives for the time step (LiftSet).
	bool BarrierContactNonsmoothTerm::lift_exact() const
	{
		return first_layer_ != nullptr && opts_.lift_model == "exact" && !lift_->ids.empty() && !problem_.uses_full_size();
	}

	void BarrierContactNonsmoothTerm::init_lift()
	{
		for (const auto &f : other_)
			if (auto e = std::dynamic_pointer_cast<ElasticForm>(f))
			{
				if (elastic_)
					log_and_throw_error("NonsmoothNewton: first_layer needs a single elastic form");
				elastic_ = e;
			}
		if (!local_ok_ || !fast_validity_ || !elastic_ || !elastic_->supports_local())
			log_and_throw_error("NonsmoothNewton: first_layer needs the local contact blocks (plain selection dof map, one FE node per collision vertex), P1 tetrahedra with a NeoHookean material and an elastic form with local derivatives");
		if (opts_.lift_model == "frozen" && opts_.lift_basis == "raw")
			log_and_throw_error("NonsmoothNewton: first_layer/lift_model frozen with lift_basis raw does not lift anything");
		first_layer_ = std::make_unique<FirstLayer>(problem_, fast_validity_, static_cast<UnitSink &>(*this));
		sync_lift();
	}

	void BarrierContactNonsmoothTerm::sync_lift()
	{
		const int nt = fast_validity_->n_tets();
		lift_mask_.assign(nt, 0);
		for (const int e : lift_->ids)
			lift_mask_[e] = 1;
		// the Jacobian bound sees the elements that are not exact (the floor and the validity stay over all)
		if (opts_.lift_model == "exact" && !lift_->ids.empty())
		{
			jratio_mask_.assign(nt, 1);
			for (const int e : lift_->ids)
				jratio_mask_[e] = 0;
		}
		else
			jratio_mask_.clear();
	}

	int BarrierContactNonsmoothTerm::add_elements(const std::vector<int> &ids)
	{
		int n = 0;
		std::vector<int> merged = lift_->ids;
		for (const int e : ids)
			if (!lift_mask_[e])
			{
				lift_mask_[e] = 1;
				merged.push_back(e);
				pending_.push_back(e);
				++n;
			}
		std::sort(merged.begin(), merged.end());
		lift_->ids = std::move(merged);
		sync_lift();
		return n;
	}

	/// The elastic part of the exact term at yf: gradient (added to grad, full dofs) and one 12 x 12 block per lifted element
	void BarrierContactNonsmoothTerm::lift_blocks(const Vec &yf, std::vector<polysolve::nonlinear::HessianBlock> &blocks, Vec &grad) const
	{
		const auto &ids = lift_->ids;
		const Eigen::MatrixXd xm = yf;
		const size_t n0 = blocks.size();
		blocks.resize(n0 + ids.size());
		struct TD
		{
			assembler::ElementAssemblyValues vals;
			ElasticForm::ElementLocal loc;
		};
		auto st = utils::create_thread_storage(TD());
		std::vector<Vec> gl(ids.size());
		utils::maybe_parallel_for(int(ids.size()), [&](int start, int end, int thread_id) {
			TD &td = utils::get_local_thread_storage(st, thread_id);
			for (int k = start; k < end; ++k)
			{
				elastic_->element_local(xm, ids[k], true, false, td.vals, td.loc);
				if (td.loc.g.size() != 12)
					throw std::runtime_error("NonsmoothNewton: first_layer needs elements with 12 local dofs");
				polysolve::nonlinear::HessianBlock &b = blocks[n0 + k];
				b.size = 12;
				b.H.topLeftCorner(12, 12) = td.loc.H;
				for (int j = 0; j < 12; ++j)
					b.indices[j] = red_of_full_[td.loc.dofs[j]];
				gl[k] = td.loc.g;
			}
		});
		for (size_t k = 0; k < ids.size(); ++k)
		{
			const auto &b = blocks[n0 + k];
			for (int j = 0; j < 12; ++j)
				if (const int r = b.indices[j]; r >= 0)
					grad[r] += gl[k][j];
		}
	}

	bool BarrierContactNonsmoothTerm::take_delta(const Vec &x, const bool projected, polysolve::nonlinear::NonsmoothDelta &delta)
	{
		std::vector<int> ids;
		ids.swap(pending_);
		if (ids.empty() || opts_.lift_model != "exact")
			return false;
		std::sort(ids.begin(), ids.end());
		const Vec xf = problem_.reduced_to_full(x);
		const int n = int(x.size());
		const Eigen::MatrixXd xm = xf;
		struct TD
		{
			assembler::ElementAssemblyValues vals;
			ElasticForm::ElementLocal loc;
		};
		auto st = utils::create_thread_storage(TD());
		std::vector<Eigen::MatrixXd> Hl(ids.size());
		std::vector<Vec> gl(ids.size());
		std::vector<std::vector<int>> dl(ids.size());
		utils::maybe_parallel_for(int(ids.size()), [&](int start, int end, int thread_id) {
			TD &td = utils::get_local_thread_storage(st, thread_id);
			for (int k = start; k < end; ++k)
			{
				elastic_->element_local(xm, ids[k], true, false, td.vals, td.loc);
				Hl[k] = projected ? Eigen::MatrixXd(ipc::project_to_psd(td.loc.H)) : td.loc.H;
				gl[k] = td.loc.g;
				dl[k] = td.loc.dofs;
			}
		});
		std::vector<Eigen::Triplet<double>> trip;
		trip.reserve(ids.size() * 144);
		Vec dg = Vec::Zero(n);
		for (size_t k = 0; k < ids.size(); ++k)
			for (int a = 0; a < 12; ++a)
			{
				const int ra = dl[k][a] >= 0 ? red_of_full_[dl[k][a]] : -1;
				if (ra < 0)
					continue;
				dg[ra] += gl[k][a];
				for (int b = 0; b < 12; ++b)
				{
					const int rb = dl[k][b] >= 0 ? red_of_full_[dl[k][b]] : -1;
					if (rb >= 0)
						trip.emplace_back(ra, rb, Hl[k](a, b));
				}
			}
		delta.dH.resize(n, n);
		delta.dH.setFromTriplets(trip.begin(), trip.end());
		delta.dg = dg;
		delta.dE = elastic_->subset_energy(xf, ids);
		delta.n = int(ids.size());
		return true;
	}

	/// The projected block of the lifted elements that B takes where the model keeps the frozen one (lift_model frozen)
	bool BarrierContactNonsmoothTerm::lift_correction(const Vec &x, const bool projected, StiffnessMatrix &dB)
	{
		if (!first_layer_ || opts_.lift_model != "frozen" || projected || lift_->ids.empty() || problem_.uses_full_size())
			return false;
		const Vec xf = problem_.reduced_to_full(x);
		StiffnessMatrix raw, proj;
		elastic_->subset_hessian(xf, lift_->ids, false, raw);
		elastic_->subset_hessian(xf, lift_->ids, true, proj);
		dB = proj - raw;
		to_reduced(dB);
		return true;
	}

	int BarrierContactNonsmoothTerm::node_of_unknown(const long index)
	{
		if (full_of_red_.empty())
		{
			int n = 0;
			for (const int r : red_of_full_)
				n = std::max(n, r + 1);
			full_of_red_.assign(n, -1);
			for (int i = 0; i < int(red_of_full_.size()); ++i)
				if (red_of_full_[i] >= 0)
					full_of_red_[red_of_full_[i]] = i;
		}
		return index >= 0 && index < long(full_of_red_.size()) && full_of_red_[index] >= 0 ? full_of_red_[index] / 3 : -1;
	}

	/// The elastic elements as units of the anchor x for the repair of B
	void BarrierContactNonsmoothTerm::repair_units(const Vec &x, std::vector<RepairUnit> &units)
	{
		if (!local_ok_)
			return;
		std::shared_ptr<ElasticForm> elastic = elastic_;
		for (const auto &f : other_)
			if (!elastic)
				elastic = std::dynamic_pointer_cast<ElasticForm>(f);
		if (!elastic || !elastic->supports_local())
			return;
		const Eigen::MatrixXd xm = problem_.reduced_to_full(x);
		assembler::ElementAssemblyValues vals;
		ElasticForm::ElementLocal loc;
		for (int e = 0; e < elastic->n_elements(); ++e)
		{
			elastic->element_local(xm, e, true, false, vals, loc);
			RepairUnit r;
			r.nd = int(loc.dofs.size());
			if (r.nd != 12)
				continue; // only linear tetrahedra
			int nodes[4];
			for (int j = 0; j < 12; ++j)
			{
				r.dof[j] = loc.dofs[j] >= 0 ? red_of_full_[loc.dofs[j]] : -1;
				if (j % 3 == 0)
					nodes[j / 3] = loc.dofs[j] / 3;
			}
			r.n_nodes = 4;
			std::copy(nodes, nodes + 4, r.node.begin());
			r.H = loc.H;
			r.frozen = !is_exact_element(e);
			units.push_back(std::move(r));
		}
	}

	polysolve::nonlinear::BasisPort *BarrierContactNonsmoothTerm::basis()
	{
		if (!basis_port_)
			basis_port_ = std::make_unique<BasisRepair>(
				[this](const Vec &x, std::vector<RepairUnit> &units) { repair_units(x, units); },
				[this](const long index) { return node_of_unknown(index); },
				[this](const Vec &x, const bool projected, StiffnessMatrix &dB) { return lift_correction(x, projected, dB); });
		return basis_port_.get();
	}

	// ------------------------------------------------------------------------------------------------------ dof maps
	// The term works in the current size of the problem (cf. NLProblem::gradient / hessian): the reduced space, or the full
	// space in the augmented Lagrangian subproblems. Without constraint forms the reduced space is the full space: the
	// reduction matrices are not set up and full-size quantities are already reduced.
	Vec BarrierContactNonsmoothTerm::to_reduced(const Vec &full) const
	{
		return problem_.full_size() != problem_.reduced_size() && !problem_.uses_full_size() ? problem_.full_to_reduced_grad(full) : full;
	}

	void BarrierContactNonsmoothTerm::to_reduced(StiffnessMatrix &hessian) const
	{
		if (problem_.full_size() != problem_.reduced_size() && !problem_.uses_full_size())
			problem_.full_hessian_to_reduced_hessian(hessian);
	}

	void BarrierContactNonsmoothTerm::init_local_map()
	{
		local_ok_ = false;
		const Vec x0 = Vec::Zero(problem_.reduced_size());
		const int nf = int(problem_.reduced_to_full(x0).size());
		bool ok = problem_.reduction_is_selection(red_of_full_) && int(red_of_full_.size()) == nf;
		if (ok)
			ok = init_node_of_vertex(nf);
		local_ok_ = ok;
		if (!local_ok_)
			logger().warn("[NonsmoothNewton] local contact blocks disabled: dof map is not a plain selection");
	}

	bool BarrierContactNonsmoothTerm::init_node_of_vertex(const int nf)
	{
		// collision vertex v follows the full vertex fv = to_full_vertex_id(v) of the collision mesh, which follows the FE
		// nodes through the displacement map (row fv); the local path needs exactly one FE node with weight 1 per row
		const int dim = 3;
		if (nf % dim != 0)
			return false;
		const int n_nodes = nf / dim;
		const int nv = int(mesh_.num_vertices());
		const int n_full = int(mesh_.full_num_vertices());
		std::vector<int> node_of_full(n_full, -1);
		if (displacement_map_ == nullptr || displacement_map_->size() == 0)
		{
			if (n_full > n_nodes)
				return false;
			for (int fv = 0; fv < n_full; ++fv)
				node_of_full[fv] = fv;
		}
		else
		{
			const Eigen::SparseMatrix<double> &T = *displacement_map_;
			if (T.rows() != n_full || T.cols() != n_nodes)
				return false;
			std::vector<int> count(n_full, 0);
			for (int col = 0; col < T.outerSize(); ++col)
				for (Eigen::SparseMatrix<double>::InnerIterator it(T, col); it; ++it)
				{
					if (it.value() == 0)
						continue;
					const int row = int(it.row());
					if (++count[row] > 1 || it.value() != 1)
						node_of_full[row] = -2; // not a single unit weight
					else
						node_of_full[row] = int(it.col());
				}
		}
		node_of_vertex_.assign(nv, -1);
		for (int v = 0; v < nv; ++v)
		{
			const int fv = int(mesh_.to_full_vertex_id(v));
			if (fv < 0 || fv >= n_full || node_of_full[fv] < 0 || node_of_full[fv] >= n_nodes)
				return false;
			node_of_vertex_[v] = node_of_full[fv];
		}
		return true;
	}

	// ---------------------------------------------------------------------------------------------------- smooth part
	void BarrierContactNonsmoothTerm::smooth_gradient(const Vec &x, Vec &grad)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		Vec gn = Vec::Zero(xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			Vec gt;
			f->first_derivative(xfull, gt);
			gn += gt;
		}
		if (lift_exact())
		{
			Vec gs;
			elastic_->subset_gradient(xfull, lift_->ids, gs);
			gn -= gs;
		}
		grad = to_reduced(gn);
		problem_.add_penalty_gradient(x, grad);
	}

	void BarrierContactNonsmoothTerm::smooth_hessian(const Vec &x, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		const Vec xfull = problem_.reduced_to_full(x);
		for (auto &f : other_)
			f->set_project_to_psd(project_to_psd);
		StiffnessMatrix h(xfull.size(), xfull.size());
		for (auto &f : other_)
		{
			if (!f->enabled())
				continue;
			StiffnessMatrix tmp;
			f->second_derivative(xfull, tmp);
			h += tmp;
		}
		if (lift_exact())
		{
			StiffnessMatrix hs;
			elastic_->subset_hessian(xfull, lift_->ids, project_to_psd, hs);
			h -= hs;
		}
		else if (first_layer_ && opts_.lift_model == "frozen_projected" && !project_to_psd && !lift_->ids.empty() && !problem_.uses_full_size())
		{
			// the model's quadratic has the projected block of the lifted elements
			StiffnessMatrix raw, proj;
			elastic_->subset_hessian(xfull, lift_->ids, false, raw);
			elastic_->subset_hessian(xfull, lift_->ids, true, proj);
			h += proj;
			h -= raw;
		}
		to_reduced(h);
		problem_.add_penalty_hessian(x, h);
		hessian = std::move(h);
	}

	void BarrierContactNonsmoothTerm::ensure_anchor(const Vec &x)
	{
		auto &a = anchor_->anchor;
		if (a.x.size() == x.size() && a.x == x)
			return;
		fast_validity_->anchor(x, problem_.reduced_to_full(x), a);
	}

	double BarrierContactNonsmoothTerm::smooth_valid_fraction(const Vec &x, const Vec &a, const Vec &b)
	{
		if (!(opts_.fast_validity && fast_validity_ != nullptr))
			return valid_fraction_native(a, b);
		ensure_anchor(x);
		return fast_validity_->valid_fraction(anchor_->anchor, a, b, [this](const Vec &y) { return problem_.reduced_to_full(y); });
	}

	double BarrierContactNonsmoothTerm::min_jacobian(const Vec &x) const
	{
		if (!fast_validity_)
			return NaN;
		return fast_validity_->min_jacobian(problem_.reduced_to_full(x));
	}

	void BarrierContactNonsmoothTerm::min_jacobians(const Vec &x, const Vec &y, const bool want_min, const bool want_ratio, double &min_j, double &min_ratio)
	{
		min_j = min_ratio = NaN;
		if (!fast_validity_ || !(want_min || want_ratio))
			return;
		if (want_ratio)
		{
			// x is the outer iterate of all trial points of an inner loop
			const Vec xf = problem_.reduced_to_full(x);
			if (!(jdet_key_.size() == xf.size() && jdet_key_ == xf))
			{
				jdet_ = fast_validity_->dets(xf);
				jdet_key_ = xf;
			}
		}
		fast_validity_->min_jacobians(jdet_, problem_.reduced_to_full(y), want_min, want_ratio, min_j, min_ratio, jratio_mask_.empty() ? nullptr : &jratio_mask_);
	}

	/// Native validity shortening (cf. polysolve compute_nan_free_step_size): halve until the step is valid and the
	/// smooth energy is finite. Returns the fraction.
	double BarrierContactNonsmoothTerm::valid_fraction_native(const Vec &a, const Vec &b)
	{
		double alpha = 1;
		for (int i = 0; i < 40; ++i, alpha *= 0.5)
		{
			const Vec y = a + alpha * (b - a);
			if (!problem_.is_step_valid(a, y))
				continue;
			const Vec yf = problem_.reduced_to_full(y);
			double e = 0;
			for (auto &f : other_)
				if (f->enabled())
					e += f->value(yf);
			if (std::isfinite(e))
				return alpha;
		}
		return 0;
	}

	void BarrierContactNonsmoothTerm::metric_diagonal(Vec &diag) const
	{
		diag = problem_.full_to_reduced_diag(mass_full_);
		if (problem_.uses_full_size())
		{
			// augmented Lagrangian subproblems: the obstacle dofs are free and massless; give them the mean mass of the
			// FE dofs, as BCLagrangianForm does for the obstacle entries of its penalty metric
			double sum = 0;
			int n = 0;
			for (int i = 0; i < diag.size(); ++i)
				if (diag[i] > 0)
				{
					sum += diag[i];
					++n;
				}
			const double mean = n ? sum / n : 1.0;
			for (int i = 0; i < diag.size(); ++i)
				if (!(diag[i] > 0))
					diag[i] = mean;
		}
	}

	// ---------------------------------------------------------------------------------------------- nonsmooth term
	void BarrierContactNonsmoothTerm::update(const Vec &y)
	{
		contact_->solution_changed(problem_.reduced_to_full(y));
	}

	double BarrierContactNonsmoothTerm::value(const Vec &y)
	{
		const Vec yf = problem_.reduced_to_full(y);
		double v = contact_->value(yf);
		if (friction_)
			v += friction_->value(yf);
		if (lift_exact())
			v += elastic_->subset_energy(yf, lift_->ids);
		return v;
	}

	void BarrierContactNonsmoothTerm::gradient(const Vec &y, Vec &grad)
	{
		const Vec yf = problem_.reduced_to_full(y);
		Vec g;
		contact_->first_derivative(yf, g);
		if (friction_)
		{
			Vec gf;
			friction_->first_derivative(yf, gf);
			g += gf;
		}
		if (lift_exact())
		{
			Vec ge;
			elastic_->subset_gradient(yf, lift_->ids, ge);
			g += ge;
		}
		grad = to_reduced(g);
	}

	void BarrierContactNonsmoothTerm::hessian(const Vec &y, const bool project_to_psd, StiffnessMatrix &hessian)
	{
		contact_->set_project_to_psd(project_to_psd);
		StiffnessMatrix K;
		const Vec yf = problem_.reduced_to_full(y);
		contact_->second_derivative(yf, K);
		if (friction_)
		{
			// the friction block is never projected (PSD for mu_s = mu_k) and leaves the form's own projection flag alone;
			// B (project_to_psd) of the first solve of a time step takes it at the velocities of the previous step
			const bool at_lag = project_to_psd && problem_.take_friction_b_lag();
			const Eigen::MatrixXd V = at_lag ? friction_->previous_surface_velocities() : friction_->compute_surface_velocities(yf);
			StiffnessMatrix Kf;
			friction_->hessian_at_velocities(V, false, Kf);
			K += Kf;
			if (project_to_psd)
			{
				++b_assemblies_;
				b_at_lag_ += at_lag;
			}
		}
		if (lift_exact())
		{
			StiffnessMatrix Ke;
			elastic_->subset_hessian(yf, lift_->ids, project_to_psd && opts_.lift_basis == "projected", Ke);
			K += Ke;
		}
		to_reduced(K);
		hessian = std::move(K);
	}

	bool BarrierContactNonsmoothTerm::hessian_blocks(const Vec &y, Vec &grad, std::vector<polysolve::nonlinear::HessianBlock> &blocks)
	{
		if (!(opts_.local_blocks && local_ok_) || problem_.uses_full_size())
			return false; // the local blocks map to the reduced dofs
		const Vec yf = problem_.reduced_to_full(y);
		const auto V = contact_->compute_displaced_surface(yf);
		const auto &cs = contact_->collision_set();
		const auto &bp = contact_->barrier_potential();
		const auto &E = mesh_.edges();
		const auto &Fc = mesh_.faces();
		const double w = contact_->weight();
		grad = Vec::Zero(y.size());
		const size_t nb = cs.size(), nf = friction_ ? friction_->friction_collision_set().size() : 0;
		blocks.clear();
		blocks.resize(nb + nf);
		// the blocks and their gradients per pair in parallel; the gradients are added below in pair order
		std::vector<ipc::VectorMax12d> gb(nb + nf);
		const auto set_indices = [&](const auto &ids, polysolve::nonlinear::HessianBlock &b) {
			for (int a = 0; a < b.size / 3; ++a)
				for (int d = 0; d < 3; ++d)
					b.indices[3 * a + d] = ids[a] >= 0 ? red_of_full_[3 * node_of_vertex_[ids[a]] + d] : -1;
		};
		utils::maybe_parallel_for(int(nb), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				const auto &col = cs[i];
				const ipc::VectorMax12d xc = col.dof(V, E, Fc);
				polysolve::nonlinear::HessianBlock &b = blocks[i];
				b.size = int(xc.size());
				gb[i] = w * bp.gradient(col, xc);
				b.H.topLeftCorner(b.size, b.size) = w * bp.hessian(col, xc, ipc::PSDProjectionMethod::NONE);
				set_indices(col.vertex_ids(E, Fc), b);
			}
		});
		if (friction_)
		{
			// one block per lagged friction pair, at the surface velocities of y (unprojected; gradient wrt x = weighted
			// gradient wrt v, Hessian wrt x = dv/dx times the weighted Hessian wrt v)
			const Eigen::MatrixXd Vv = friction_->compute_surface_velocities(yf);
			const auto &fcs = friction_->friction_collision_set();
			const auto &fp = friction_->friction_potential();
			const double cf = friction_->derivative_factor(), dvdx = friction_->dv_dx();
			utils::maybe_parallel_for(int(nf), [&](int start, int end, int /*thread_id*/) {
				for (size_t i = start; i < size_t(end); ++i)
				{
					const auto &col = fcs[i];
					const ipc::VectorMax12d vc = col.dof(Vv, E, Fc);
					polysolve::nonlinear::HessianBlock &b = blocks[nb + i];
					b.size = int(vc.size());
					gb[nb + i] = cf * fp.gradient(col, vc);
					b.H.topLeftCorner(b.size, b.size) = (cf * dvdx) * fp.hessian(col, vc, ipc::PSDProjectionMethod::NONE);
					set_indices(col.vertex_ids(E, Fc), b);
				}
			});
		}
		for (size_t i = 0; i < nb + nf; ++i)
		{
			const auto &b = blocks[i];
			for (int j = 0; j < b.size; ++j)
				if (const int r = b.indices[j]; r >= 0)
					grad[r] += gb[i][j];
		}
		if (lift_exact())
			lift_blocks(yf, blocks, grad);
		return true;
	}

	int BarrierContactNonsmoothTerm::active_size() const
	{
		// lagged friction pairs are active terms too (a path is taken with friction even without barrier contacts)
		return int(contact_->collision_set().size()) + (friction_ ? int(friction_->friction_collision_set().size()) : 0);
	}

	void BarrierContactNonsmoothTerm::active_distances(const Vec &a, const Vec &b, Vec &distance, Vec &rate)
	{
		const auto Va = contact_->compute_displaced_surface(problem_.reduced_to_full(a));
		const auto Vb = contact_->compute_displaced_surface(problem_.reduced_to_full(b));
		const auto &cs = contact_->collision_set();
		const auto &Em = mesh_.edges();
		const auto &Fm = mesh_.faces();
		// per pair in parallel, then the pairs with a positive distance in order
		const size_t n = cs.size();
		std::vector<double> d_all(n), r_all(n);
		std::vector<char> positive(n, 0);
		utils::maybe_parallel_for(int(n), [&](int start, int end, int /*thread_id*/) {
			for (size_t i = start; i < size_t(end); ++i)
			{
				const auto &c = cs[i];
				const Vec xa = c.dof(Va, Em, Fm), xb = c.dof(Vb, Em, Fm);
				const double d2 = c.compute_distance(xa);
				if (!(d2 > 0))
					continue;
				const double dist = std::sqrt(d2);
				d_all[i] = dist;
				r_all[i] = c.compute_distance_gradient(xa).dot(xb - xa) / (2 * dist); // d'(0) along the segment
				positive[i] = 1;
			}
		});
		std::vector<double> dv, rv;
		dv.reserve(n);
		rv.reserve(n);
		for (size_t i = 0; i < n; ++i)
			if (positive[i])
			{
				dv.push_back(d_all[i]);
				rv.push_back(r_all[i]);
			}
		distance = Eigen::Map<const Vec>(dv.data(), Eigen::Index(dv.size()));
		rate = Eigen::Map<const Vec>(rv.data(), Eigen::Index(rv.size()));
	}

	double BarrierContactNonsmoothTerm::step_length(const Vec &a, const Vec &b)
	{
		const auto Va = contact_->compute_displaced_surface(problem_.reduced_to_full(a));
		const auto Vb = contact_->compute_displaced_surface(problem_.reduced_to_full(b));
		return (Vb - Va).rowwise().norm().maxCoeff();
	}

	double BarrierContactNonsmoothTerm::activation_distance() const
	{
		return contact_->dhat();
	}

	void BarrierContactNonsmoothTerm::segment_begin(const Vec &a, const Vec &b)
	{
		problem_.line_search_begin(a, b);
		if (opts_.candidate_permutation > 0)
			candidate_permute(contact_->cached_candidates(), opts_.candidate_permutation);
	}

	bool BarrierContactNonsmoothTerm::certified_max_step_size() const
	{
		return opts_.certificate && !problem_.is_residual() && contact_->uses_cached_candidates();
	}

	double BarrierContactNonsmoothTerm::max_step_size(const Vec &x0, const Vec &x1)
	{
		return certifier_->max_step_size(x0, x1);
	}

	json BarrierContactNonsmoothTerm::stats() const
	{
		json s = {{"certified_pairs", certifier_->n_certified},
				  {"ti_pairs", certifier_->n_ti},
				  {"local_contact_blocks", local_ok_}};
		if (friction_)
		{
			s["friction_pairs"] = friction_->friction_collision_set().size();
			s["friction_b_assemblies"] = b_assemblies_;
			s["friction_b_at_lag"] = b_at_lag_;
		}
		return s;
	}
} // namespace polyfem::solver
