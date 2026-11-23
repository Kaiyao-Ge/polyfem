set(SOURCES
	ALSolver.cpp
	ALSolver.hpp
	BarrierContactNonsmoothTerm.cpp
	BarrierContactNonsmoothTerm.hpp
	BilateralSegmentCertifier.cpp
	BilateralSegmentCertifier.hpp
	FastTetValidity.cpp
	FastTetValidity.hpp
	FullNLProblem.cpp
	FullNLProblem.hpp
	NonsmoothTerm.hpp
	NLProblem.cpp
	NLProblem.hpp
	NLHomoProblem.cpp
	NLHomoProblem.hpp
	OperatorSplittingSolver.hpp
	OperatorSplittingSolver.cpp
	nonsmooth/FirstLayer.hpp
	nonsmooth/FirstLayer.cpp
	nonsmooth/blind/LocalTerms.hpp
	nonsmooth/blind/LocalTerms.cpp
	nonsmooth/blind/UnitCache.hpp
	nonsmooth/blind/UnitCache.cpp
	nonsmooth/blind/BlindSession.hpp
	nonsmooth/blind/BlindSession.cpp
	nonsmooth/blind/UnitSelector.hpp
	nonsmooth/blind/UnitSelector.cpp
	nonsmooth/blind/BlindUnitTerm.hpp
	nonsmooth/blind/BlindUnitTerm.cpp
	SolveData.cpp
	SolveData.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(forms)
