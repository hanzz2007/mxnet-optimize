/*!
 * Copyright (c) 2017 by iFLYTEK Ltd.
 * \file lstm.cc
 * \brief
 * \author qiuhan
*/

#include "./lstm-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(LSTMParam);

template<>
Operator* CreateOp<cpu>(LSTMParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSTMOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *LSTMProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(LSTM, LSTMProp)
.describe(R"code(Compute LSTM
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the LSTMOp.")
.add_argument("weight_i2h", "NDArray-or-Symbol", "i2h Weight matrix.")
.add_argument("weight_h2h", "NDArray-or-Symbol", "h2h Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(LSTMParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
