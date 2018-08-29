/*!
 * Copyright (c) 2017 iFLYTEK Ltd.
 * \file lstm-inl.h
 * \brief 没有peephole的三个门的标准LSTM实现
 *		  基于mshadow实现cpu和gpu跨平台计算, 也许效率比cuda实现要低效?
 * \author qiuhan@iflytek.com
*/
#ifndef MXNET_OPERATOR_LSTM_INL_H_
#define MXNET_OPERATOR_LSTM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace lstm_enum {
  enum LSTMOpInputs {kData, kWi2h, kWh2h, kBias};
  enum LSTMOpOutputs {kOut};
  enum LSTMOpResource {kTempSpace};
}


struct LSTMParam : public dmlc::Parameter<LSTMParam> {
  uint32_t num_output;
  float clipping_threshold;
  bool bidirectional;

  DMLC_DECLARE_PARAMETER(LSTMParam) {
    DMLC_DECLARE_FIELD(num_output)
    .describe(" number of hidden units");

    DMLC_DECLARE_FIELD(clipping_threshold)
    .describe("clipping threshold");

    DMLC_DECLARE_FIELD(bidirectional).set_default(false)
    .describe("whether to use bidirectional recurrent layers");
  }
};

#define BLOB_CHAIN_BEGIN(NAME, SHAPE) \
    TBlob NAME##_blob (workspace.dptr_, SHAPE, xpu::kDevMask, DataType<DType>::kFlag); \
	Tensor<xpu, SHAPE##.kDimension, DType> NAME = NAME##_blob.get_with_shape<xpu, SHAPE##.kDimension, DType>(SHAPE, s);

#define BLOB_CHAIN_END()

#define DECLARE_BLOB_CHAIN(NAME, SHAPE, PREV) \
		TBlob NAME##_blob(PREV##_blob.dptr<DType>() + PREV##_blob.Size(), SHAPE, xpu::kDevMask, DataType<DType>::kFlag); \
		Tensor<xpu, SHAPE##.kDimension, DType> NAME = NAME##_blob.get_with_shape<xpu, SHAPE##.kDimension, DType>(SHAPE, s);

template<typename xpu, typename DType>
class LSTMOp : public Operator {
 public:
  explicit LSTMOp(LSTMParam p) {
	  param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

	CHECK_EQ(req[lstm_enum::kOut], kWriteTo);
	const size_t expected = 4;
	CHECK_EQ(in_data.size(), expected);
	CHECK_EQ(out_data.size(), 1);
	
	Stream<xpu>* s = ctx.get_stream<xpu>();

	// (T, N, I)
	TBlob data_blob = in_data[lstm_enum::kData];
	Tensor<xpu, 3, DType> bottom_data = data_blob.get<xpu, 3, DType>(s);

	// (T, N, H)
	TBlob out_blob = out_data[lstm_enum::kOut];
	Tensor<xpu, 3, DType> top_data = out_blob.get<xpu, 3, DType>(s);

	const int T = data_blob.size(0);
	const int N = data_blob.size(1);
	const int I = data_blob.size(2);
	const int H = param_.num_output;

	CHECK_EQ(out_blob.shape_, Shape3(T, N, H));

	// (4 * H, I)
	TBlob weight_i_blob = in_data[lstm_enum::kWi2h];
	Tensor<xpu, 2, DType> weight_i = weight_i_blob.get<xpu, 2, DType>(s);

	// (4 * H, H)
	TBlob weight_h_blob = in_data[lstm_enum::kWh2h];
	Tensor<xpu, 2, DType> weight_h = weight_h_blob.get<xpu, 2, DType>(s);

	// (4 * H)
	TBlob bias_blob = in_data[lstm_enum::kBias];
	Tensor<xpu, 1, DType> bias = bias_blob.get<xpu, 1, DType>(s);
	
	auto cell_shape = Shape2(N, H);
	auto gate_shape = Shape3(N, 4, H);
	auto pre_gate_shape = Shape4(T, N, 4, H);
	auto top_shape = Shape3(T, N, H);
	auto bias_shape = Shape2(N * T, 4 * H);

	int work_space_size = 
		cell_shape.Size() * 5  // c_0, h_0, c_T, h_T, h_to_h
		+ gate_shape.Size() // h_to_gate
		+ pre_gate_shape.Size() * 2 // pre_gate, gate
		+ top_shape.Size() // cell
		/*+ bias_shape.Size()*/; // bias

	Tensor<xpu, 1, DType> workspace = ctx.requested[lstm_enum::kTempSpace]
		.get_space_typed<xpu, 1, DType>(Shape1(work_space_size), s);

		// (N, H)
	// 	TBlob c_0_blob(workspace.dptr_, cell_shape, xpu::kDevMask, DataType<DType>::kFlag);
// 	auto c_0 = c_0_blob.get_with_shape<xpu, cell_shape.kDimension, DType>(cell_shape, s);
	BLOB_CHAIN_BEGIN(c_0, cell_shape);
	DECLARE_BLOB_CHAIN(h_0, cell_shape, c_0);
	DECLARE_BLOB_CHAIN(c_T, cell_shape, h_0);
	DECLARE_BLOB_CHAIN(h_T, cell_shape, c_T);

	// hiden to hiden
	DECLARE_BLOB_CHAIN(h_to_gate, gate_shape, h_T);

	DECLARE_BLOB_CHAIN(pre_gate, pre_gate_shape, h_to_gate);
	DECLARE_BLOB_CHAIN(gate, pre_gate_shape, pre_gate);

	DECLARE_BLOB_CHAIN(cell, top_shape, gate);
	BLOB_CHAIN_END()

	c_0 = 0;
	h_0 = 0;

	auto pre_gate_2d = pre_gate_blob.get_with_shape<xpu, 2, DType>(Shape2(T * N, 4 * H), s);
	pre_gate_2d = dot(bottom_data.FlatTo2D(), weight_i.T());
	pre_gate_2d += repmat(bias, T * N);

	for (int t = 0; t < T; ++t)
	{
		// (N, H)
		auto h_t = top_data[t];
		// (N, H)
		auto c_t = cell[t];

		// (N, 4 * H)
		auto pre_gate_t_2d = TBlob(pre_gate[t])
			.get_with_shape<xpu, 2, DType>(Shape2(N, 4 * H), s);
		auto h_to_gate_t_2d = TBlob(h_to_gate)
			.get_with_shape<xpu, 2, DType>(Shape2(N, 4 * H), s);

		// (N, H)
		auto h_t_1 = t > 0 ? top_data[t - 1] : h_0;
		auto c_t_1 = t > 0 ? cell[t - 1] : c_0;

		// h(t -1) * W_{hh} => (N, 4 * H)
		h_to_gate_t_2d = dot(h_t_1, weight_h.T());

		const bool cont = (t > 0);
		// x(t) * W_{xh} + bias + h(t-1) * W_{hh}
		if (cont) {
			pre_gate_t_2d += h_to_gate_t_2d;
		}

#if 0
		// (N, 4, H) => (4, N, H)
		Tensor<xpu, 3, DType> gate_t; gate_t = swapaxis<1, 0>(gate[t]);
		// (N, 4, H) => (4, N, H)
		Tensor<xpu, 3, DType> pre_gate_t; pre_gate_t = swapaxis<1, 0>(pre_gate[t]);

#if 1
		// i f o g
		gate_t.Slice(0, 3) = F<mshadow_op::sigmoid>(pre_gate_t.Slice(0, 3));
#else
		gate_t[0] = F<mshadow_op::sigmoid>(pre_gate_t[0]);
		gate_t[1] = cont ? F<mshadow_op::sigmoid>(pre_gate_t[1]) : c_0;
		gate_t[2] = F<mshadow_op::sigmoid>(pre_gate_t[2]);
#endif
		gate_t[3] = F<mshadow_op::tanh>(pre_gate_t[3]);

		// Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
		if (cont)
			c_t = gate_t[1] * c_t_1;
		else
			c_t = c_0;
		c_t += gate_t[0] * gate_t[3];
		h_t = gate_t[2] * F<mshadow_op::tanh>(c_t);

#else
		Tensor<xpu, 3, DType> gate_t = gate[t];
		Tensor<xpu, 3, DType> pre_gate_t = pre_gate[t];

		for (int n = 0; n < N; ++n)
		{
			// (4, H)
			Tensor<xpu, 2, DType> gate_t_n = gate_t[n];
			Tensor<xpu, 2, DType> pre_gate_t_n = pre_gate_t[n];

			// i f o g
			gate_t_n[0] = F<mshadow_op::sigmoid>(pre_gate_t_n[0]);
			if (cont)
				gate_t_n[1] = F<mshadow_op::sigmoid>(pre_gate_t_n[1]);
			else
				gate_t_n[1] = 0;
			gate_t_n[2] = F<mshadow_op::sigmoid>(pre_gate_t_n[2]);
			gate_t_n[3] = F<mshadow_op::tanh>(pre_gate_t_n[3]);

			// Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
			c_t[n] = gate_t_n[1] * c_t_1[n] + gate_t_n[0] * gate_t_n[3];
			h_t[n] = gate_t_n[2] * F<mshadow_op::tanh>(c_t[n]);
		}
#endif
	}
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(sbodenstein): add MShadow implementation
  }

 private:
  LSTMParam param_;
};  // class LSTMOp

template<typename xpu>
Operator* CreateOp(LSTMParam param, int dtype);

#if DMLC_USE_CXX11
class LSTMProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
      return {"data", "weight_i2h", "weight_h2h", "bias"};
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> outputs = {"output"};
    return outputs;
  }

  int NumOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
	  std::vector<TShape> *out_shape,
	  std::vector<TShape> *aux_shape) const override {
	  using namespace mshadow;
	  CHECK_EQ(in_shape->size(), 4U) << "Input:[data, weight_i2h, weight_h2h, bias]";
	  const TShape &dshape = (*in_shape)[lstm_enum::kData];
	  if (dshape.ndim() == 0) return false;
	  CHECK_EQ(dshape.ndim(), 3U) \
		  << "Input data should be rank-3 tensor of dim [sequence length(T), batch size(N), input size(I)]";

	  out_shape->resize(1, TShape());

	  // data: [sequence len, batch, input dimension]
	  int T = dshape[0];
	  int N = dshape[1];
	  int I = dshape[2];
	  int H = param_.num_output;

	  SHAPE_ASSIGN_CHECK(*in_shape,
		  lstm_enum::kWi2h,
		  Shape2(4 * H, I));
	  SHAPE_ASSIGN_CHECK(*in_shape,
		  lstm_enum::kWh2h,
		  Shape2(4 * H, H));
	  SHAPE_ASSIGN_CHECK(*in_shape,
		  lstm_enum::kBias,
		  Shape1(4 * H));
	  SHAPE_ASSIGN_CHECK(*out_shape,
		  lstm_enum::kOut,
		  Shape3(T, N, H));

	  return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new LSTMProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "LSTM";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep = {in_data[lstm_enum::kData], in_data[lstm_enum::kWi2h], in_data[lstm_enum::kBias],
        in_data[lstm_enum::kWh2h], out_data[lstm_enum::kOut], out_grad[lstm_enum::kOut]};
    return dep;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  LSTMParam param_;
};  // class LSTMProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_LSTM_INL_H_
