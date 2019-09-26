#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include "swaption_exposure.hpp"

namespace py = pybind11;

using namespace py::literals;

PYBIND11_MODULE(swaption_pricing, m)
{
    auto swaption_exposure_noreturn = [](const std::vector<value_t>& swap_exposure, const std::vector<value_t>& sdf, const std::vector<int>& ex_indices, int maturity_index) {
        void(swaption_exposure(swap_exposure, sdf, ex_indices, maturity_index));
    };

	py::class_<inputs_holder>(m, "RollbackInputs")
		.def_readonly("swap_exposure", &inputs_holder::swap_exposure)
		.def_readonly("sdf", &inputs_holder::sdf)
		.def_readonly("ex_indices", &inputs_holder::ex_indices)
		.def_readonly("maturity_index", &inputs_holder::maturity_index);

	py::class_<outputs_holder>(m, "RollbackOutputs")
		.def_readonly("swaption_exposure", &outputs_holder::swaption_exposure)
		.def_readonly("npv", &outputs_holder::npv);

    m.def("setup", &setup, "paths"_a, "grid_size"_a, "ex_times_size"_a, "seed"_a);
    m.def("swaption_exposure", py::overload_cast<const inputs_holder&>(&swaption_exposure), "inputs"_a);
}
