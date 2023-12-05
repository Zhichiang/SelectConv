#include <torch/extension.h>
#include <vector>
#include "select_conv_cuda.hpp"
using namespace at;

// 前向传播 output = x + 2*y
Tensor add_forward_cpu(const Tensor& x,const Tensor& y) {
    if (x.device().is_cuda()) {
#ifdef WITH_CUDA // 这个宏将由setuptools定义，详见setuptools内容，总之可以用这个宏来区分是否编译cuda代码
        // add_forward_cuda(x,y); // cuda编程，先按下不表
#else
        AT_ERROR("not compile with GPU support");
#endif
    }else{
        AT_ASSERTM(x.sizes() == y.sizes(), "x must be the same size as y");
        Tensor z = zeros(x.sizes());
        z = 2 * x + y;
        return z;
    }
}

std::vector<torch::Tensor> add_backward_cpu(const Tensor& gradOutput) { // gradOutput是前向传播的输出tensor的梯度
    if (gradOutput.device().is_cuda()) {
#ifdef WITH_CUDA 
        // add_backward_cuda(x,y); // cuda编程，先按下不表
        Tensor gradOutputX = 2 * gradOutput * ones(gradOutput.sizes());
        Tensor gradOutputY = gradOutput * ones(gradOutput.sizes());
        return {gradOutputX, gradOutputY};
#else
        AT_ERROR("not compile with GPU support");
#endif
    } else {
        Tensor gradOutputX = 2 * gradOutput * ones(gradOutput.sizes());
        Tensor gradOutputY = gradOutput * ones(gradOutput.sizes());
        return {gradOutputX, gradOutputY};
    }
}

// void select_conv_forward(
//     Tensor input, Tensor weight,
//     Tensor output, Tensor columns, Tensor ones, int kW,
//     int kH, int dW, int dH, int padW, int padH,
//     int dilationW, int dilationH, int group, int im2col_step);

// void select_conv_backward_input(
//     Tensor input, Tensor gradOutput, Tensor gradInput,
//     Tensor weight, Tensor columns, int kW, int kH,
//     int dW, int dH, int padW, int padH,
//     int dilationW, int dilationH, int group, int im2col_step);

// void select_conv_backward_parameters(
//     Tensor input, Tensor gradOutput, Tensor gradWeight,
//     Tensor columns, Tensor ones, int kW,
//     int kH, int dW, int dH, int padW, int padH,
//     int dilationW, int dilationH, int group,
//     float scale, int im2col_step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &select_conv_forward, "select conv forward cuda",
        py::arg("input"), py::arg("weight"), py::arg("output"), py::arg("columns"),
        py::arg("ones"), py::arg("kW"), py::arg("kH"), py::arg("dW"),
        py::arg("dH"), py::arg("padW"), py::arg("padH"), py::arg("dilationW"),
        py::arg("dilationH"), py::arg("group"), py::arg("im2col_step")
    );
    m.def(
        "backward_input", &select_conv_backward_input,
        "select conv backward for input cuda",
        py::arg("input"), py::arg("gradOutput"), py::arg("gradInput"), py::arg("weight"),
        py::arg("columns"), py::arg("kW"), py::arg("kH"), py::arg("dW"),
        py::arg("dH"), py::arg("padW"), py::arg("padH"), py::arg("dilationW"),
        py::arg("dilationH"), py::arg("group"), py::arg("im2col_step")
    );
    m.def(
        "backward_param", &select_conv_backward_parameters,
        "select conv backward for parameters cuda",
        py::arg("input"), py::arg("gradOutput"), py::arg("gradWeight"), py::arg("columns"),
        py::arg("ones"), py::arg("kW"), py::arg("kH"), py::arg("dW"),
        py::arg("dH"), py::arg("padW"), py::arg("padH"), py::arg("dilationW"),
        py::arg("dilationH"), py::arg("group"), py::arg("scale"), py::arg("im2col_step")
    );
}
