#ifndef TASK_H
#define TASK_H
#include "srsran/common/byte_buffer.h"
#include "srsran/common/phy_cfg_nr.h"
#include <memory>
#include <vector>
struct Task {
  std::shared_ptr<std::vector<cf_t> > buffer;
  std::shared_ptr<std::vector<cf_t> > last_slot;
  uint32_t                            slot_idx;
  srsran_timestamp_t                  ts;
  uint32_t                            task_idx;
};
#endif // TASK_H