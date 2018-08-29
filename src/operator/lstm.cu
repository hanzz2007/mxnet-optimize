/*!
 * Copyright (c) 2017 by iFLYTEK Ltd.
 * \file lstm.cc
 * \brief
 * \author qiuhan
*/

#include "./lstm-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(LSTMParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSTMOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet
