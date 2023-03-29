#include "op_declare.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

      m.def("calculate_force",
            &torch_launch_calculate_force,
            "calculate force kernel warpper");

      m.def("calculate_force_grad",
            &torch_launch_calculate_force_grad,
            "calculate force grad kernel warpper");

      m.def("calculate_virial_force",
            &torch_launch_calculate_virial_force,
            "calculate virial force kernel warpper");

      m.def("calculate_virial_force_grad",
            &torch_launch_calculate_virial_force_grad,
            "calculate virial force grad kernel warpper");

      m.def("calculate_DR",
            &torch_launch_calculate_DR,
            "calculate DR kernel warpper");

      m.def("calculate_DR_grad",
            &torch_launch_calculate_DR_grad,
            "calculate DR grad kernel warpper");

      m.def("calculate_DR_second_grad",
            &torch_launch_calculate_DR_second_grad,
            "calculate DR second grad kernel warpper");

      m.def("matmul_bias_tanh",
            &torch_launch_matmul_bias_tanh,
            "matmul bias tanh kernel warpper");

      m.def("matmul_bias",
            &torch_launch_matmul_bias,
            "matmul bias tanh kernel warpper");

      m.def("matmul",
            &torch_launch_matmul,
            "matmul kernel warpper");
}

TORCH_LIBRARY(op, m)
{
      m.def("calculate_force", torch_launch_calculate_force);

      m.def("calculate_force_grad", torch_launch_calculate_force_grad);

      m.def("calculate_virial_force", torch_launch_calculate_virial_force);

      m.def("calculate_virial_force_grad", torch_launch_calculate_virial_force_grad);

      m.def("calculate_DR", torch_launch_calculate_DR);

      m.def("calculate_DR_grad", torch_launch_calculate_DR_grad);

      m.def("calculate_DR_second_grad", torch_launch_calculate_DR_second_grad);

      m.def("matmul_bias_tanh", torch_launch_matmul_bias_tanh);

      m.def("matmul_bias", torch_launch_matmul_bias);

      m.def("matmul", torch_launch_matmul);
}