#include "gemmini.h"
#include "mmu.h"
#include "trap.h"
#include <stdexcept>
#include <iostream>
#include <assert.h>

REGISTER_EXTENSION(gemmini, []() { printf("REGISTERING GEMMINI-BETA ISA\n\n"); return new gemmini_t; })

void gemmini_state_t::reset()
{
  enable = true;

  a_addr = b_addr = c_addr = d_addr = 0;
  m = n = k = 0;
  
  mode = OS;
  act = NONE;
  acc_shift = 0;
  sys_shift = 0;
  relu6_shift = 0;
  output_sp_addr = 0;
  load_stride = dim * sizeof(input_t);
  store_stride = dim * sizeof(input_t);
  spad = new std::vector<std::vector<input_t>>(sp_matrices*dim, std::vector<input_t>(dim));
  for (size_t row = 0; row < sp_matrices*dim; ++row) {
    for (size_t elem = 0; elem < dim; ++elem) {
      spad->at(row).at(elem) = 0;
    }
  }
  pe_state = new std::vector<std::vector<accum_t>>(dim, std::vector<accum_t>(dim));
  accumulator = new std::vector<std::vector<accum_t>>(accum_rows, std::vector<accum_t>(dim));
  for (size_t row = 0; row < accum_rows; ++row) {
    for (size_t elem = 0; elem < dim; ++elem) {
      accumulator->at(row).at(elem) = 0;
    }
  }

  printf("Gemmini extension configured with:\n");
  printf("    dim = %u\n", dim);
}

void gemmini_t::reset() {
  gemmini_state.reset();
}

template <class T>
T gemmini_t::read_from_dram(reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load_uint8(addr + byte_idx) << (byte_idx*8);
  }
  return value;
}

template <class T>
void gemmini_t::write_to_dram(reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store_uint8(addr + byte_idx, (data >> (byte_idx*8)) & 0xFF);
  }
}

void gemmini_t::setmode(reg_t rs1, reg_t rs2) {
  if ((rs1 & 0b11) == 0) { // rs1[1:0] == 2'b00, config_ex, configure execute pipeline
    gemmini_state_t::Dataflow new_mode;
    gemmini_state_t::Activation new_act;
    reg_t new_acc_shift, new_sys_shift, new_relu6_shift;

    auto rs1_2 = (rs1 >> 2) & 0b1; // extract rs1[2], 0 = output stationary, 1 = weight stationary
    if (rs1_2 == 0) {
      new_mode = gemmini_state_t::OS;
    } else {
      new_mode = gemmini_state_t::WS;
    }

    auto rs1_4_3 = (rs1 >> 3) & 0b11; // extract rs1[4:3], 0 = no activation, 1 = ReLU, 2 = ReLU6
    if (rs1_4_3 == 0) {
      new_act = gemmini_state_t::NONE;
    } else if (rs1_4_3 == 1) {
      new_act = gemmini_state_t::RELU;
    } else if (rs1_4_3 == 2) {
      new_act = gemmini_state_t::RELU6;
    } else {
      assert(false);
    }

    new_acc_shift = (rs1 >> 32) & 0xFFFFFFFF;
    new_sys_shift = (rs2) & 0xFFFFFFFF;
    new_relu6_shift = (rs2 >> 32) & 0xFFFFFFFF;

    dprintf("GEMMINI: config_ex - set dataflow mode from %d to %d\n", gemmini_state.mode, new_mode);
    dprintf("GEMMINI: config_ex - set activation function from %d to %d\n", gemmini_state.act, new_act);
    dprintf("GEMMINI: config_ex - set acc_shift from %lu to %lu\n", gemmini_state.acc_shift, new_acc_shift);
    dprintf("GEMMINI: config_ex - set sys_shift from %lu to %lu\n", gemmini_state.sys_shift, new_sys_shift);
    dprintf("GEMMINI: config_ex - set relu6_shift from %lu to %lu\n", gemmini_state.relu6_shift, new_relu6_shift);

    gemmini_state.mode = new_mode;
    gemmini_state.act = new_act;

    assert(new_acc_shift >= 0 && new_acc_shift < sizeof(accum_t)*8);
    assert(new_sys_shift >= 0 && new_sys_shift < sizeof(output_t)*8);
    assert(new_relu6_shift >= 0);
    gemmini_state.acc_shift = new_acc_shift;
    gemmini_state.sys_shift = new_sys_shift;
    gemmini_state.relu6_shift = new_relu6_shift;
  } else if ((rs1 & 0b11) == 1) { // rs1[1:0] == 2'b01, config_mvin, configure load pipeline
    dprintf("GEMMINI: config_mvin - set load stride from %lu to %lu\n", gemmini_state.load_stride, rs2);
    gemmini_state.load_stride = rs2;
  } else if ((rs1 & 0b11) == 2) { // rs1[1:0] == 2'b10, config_mvout, configure store pipeline
    dprintf("GEMMINI: config_mvout - set store stride from %lu to %lu\n", gemmini_state.store_stride, rs2);
    gemmini_state.store_stride = rs2;
  }
}

void gemmini_t::compute(reg_t a_addr, reg_t bd_addr, bool preload) {
  auto a_addr_real = static_cast<uint32_t>(a_addr & 0xFFFFFFFF);
  auto bd_addr_real = static_cast<uint32_t>(bd_addr & 0xFFFFFFFF);

  auto A = new std::vector<std::vector<input_t>>(gemmini_state.m, std::vector<input_t>(gemmini_state.n));
  auto B = new std::vector<std::vector<input_t>>(gemmini_state.n, std::vector<input_t>(gemmini_state.k));
  auto D = new std::vector<std::vector<input_t>>(gemmini_state.m, std::vector<input_t>(gemmini_state.k));
  auto result = new std::vector<std::vector<input_t>>(gemmini_state.m, std::vector<input_t>(gemmini_state.k, 0));

  // Load from memory 
  for (size_t i = 0; i < gemmini_state.m; i++) {
    auto const dram_row_addr = gemmini_state.a_addr + i*sizeof(input_t)*gemmini_state.n;
    for (size_t j = 0; j < gemmini_state.n; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(input_t);
      A->at(i).at(j) = read_from_dram<input_t>(dram_byte_addr);
    }
  }
  for (size_t i = 0; i < gemmini_state.n; i++) {
    auto const dram_row_addr = gemmini_state.b_addr + i*sizeof(input_t)*gemmini_state.k;
    for (size_t j = 0; j < gemmini_state.k; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(input_t);
      B->at(i).at(j) = read_from_dram<input_t>(dram_byte_addr);
    }
  }
  for (size_t i = 0; i < gemmini_state.m; i++) {
    auto const dram_row_addr = gemmini_state.d_addr + i*sizeof(input_t)*gemmini_state.k;
    for (size_t j = 0; j < gemmini_state.k; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(input_t);
      D->at(i).at(j) = read_from_dram<input_t>(dram_byte_addr);
    }
  }

  // Multiply & apply activation
  for (size_t x=0; x<gemmini_state.m; x++) {
    for (size_t j=0; j<gemmini_state.k; j++) {
      accum_t value = D->at(x).at(j);
      for (size_t k=0; k<gemmini_state.n; k++) {
        value += A->at(x).at(k) * B->at(k).at(j);
      }
      input_t shifted = gemmini_state.mode == gemmini_state_t::OS ?
                             rounding_saturating_shift<input_t>(value, gemmini_state.sys_shift) :
                             rounding_saturating_shift<input_t>(value, 0);
      input_t activated = apply_activation(shifted);
      result->at(x).at(j) = activated;
    }
  }
  
  // Write back to memory
  for (size_t i = 0; i < gemmini_state.m; i++) {
    auto const dram_row_addr = gemmini_state.c_addr + i*sizeof(input_t)*gemmini_state.k;
    for (size_t j = 0; j < gemmini_state.k; j++) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(input_t);
      write_to_dram<input_t>(dram_byte_addr, result->at(i).at(j));
    }
  } 
}

reg_t gemmini_t::custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  insn.funct = (insn.funct & 0b1111); // Strip the dependency bits from the funct field
  
  // FIXME: check we have that fourth bit available
  // printf("GEMMINI INSTRUCTION: %d\n", insn.funct);

  if (insn.funct == mvin_funct)
    printf("GEMMINI: deprecated `mvin` instruction will be ignored\n");
  else if (insn.funct == mvout_funct)
    printf("GEMMINI: deprecated `mvout` instruction will be ignored\n");
  else if (insn.funct == preload_funct)
    printf("GEMMINI: deprecated `preload` instruction will be ignored \n");
  else if (insn.funct == setmode_funct)
    setmode(xs1, xs2);
  else if (insn.funct == compute_preloaded_funct)
    compute(xs1, xs2, true);
  else if (insn.funct == compute_accumulated_funct)
    // FIXME: whether to keep, adapt, or drop "compute accumulated"
    compute(xs1, xs2, false);
  else if (insn.funct == flush_funct) {
    printf("GEMMINI: deprecated `flush` instruction will be ignored\n");
  } else if (insn.funct == config_addr_AB_funct) {
    gemmini_state.a_addr = xs1;
    gemmini_state.b_addr = xs2;
  } else if (insn.funct == config_addr_CD_funct ){
    gemmini_state.c_addr = xs1;
    gemmini_state.d_addr = xs2;
  } else if (insn.funct == config_size0_funct ){
    gemmini_state.m = xs1;
    gemmini_state.n = xs2;
  } else if (insn.funct == config_size1_funct ){
    gemmini_state.k = xs1;
  } else if (insn.funct == config_reset) {
    reset();
  }
  else {
    printf("GEMMINI: encountered unknown instruction with funct: %d\n", insn.funct);
    illegal_instruction();
  }
  return 0;
}

// Applying activation from PE post-shifted output to scratchpad (for OS dataflow)
// or from accumulator to DRAM (after shifting, for WS dataflow)
input_t gemmini_t::apply_activation(input_t value) {
  if (gemmini_state.act == gemmini_state_t::RELU) {
    return value > 0 ? static_cast<input_t>(value) : static_cast<input_t>(0);
  } else if (gemmini_state.act == gemmini_state_t::RELU6) {
    auto positive = value > 0 ? value : static_cast<input_t>(0);
    return value > (6 << gemmini_state.relu6_shift) ? static_cast<input_t>(6 << gemmini_state.relu6_shift) : positive;
  } else if (gemmini_state.act == gemmini_state_t::NONE) {
    return static_cast<input_t>(value);
  } else assert(false);
}

template <class T>
T gemmini_t::rounding_saturating_shift(accum_t value, uint64_t shift) {
  // Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
  int r = (shift == 0 ? 0 : ((value >> (shift-1)) & 1)) &
       (((shift <= 1 ? 0 : (value & ((1 << (shift-1)) - 1))) != 0) | ((value >> shift) & 1));
  accum_t shifted = (value >> shift) + r;

  // Saturate and cast element
  auto elem_t_max = std::numeric_limits<T>::max();
  auto elem_t_min = std::numeric_limits<T>::min();
  int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
  return static_cast<T>(elem);
}
