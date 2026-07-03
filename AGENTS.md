# RRC Reconfiguration Data Export to InfluxDB

## What was done

Added a new InfluxDB measurement `rrc_reconfig` that exports data from successfully decoded
RRC Reconfiguration messages (only when PDCP encryption is not active — i.e., before
Security Mode Command completes, or when NEA0 is used).

The export follows the same pattern as `mib`, `prach_cfg`, `cell_config`, `band_report`,
and `channel_config`.

## Data flow

```
gNB → PDSCH → UE DL Worker → handle_dlsch()
    ↓ (decodes RRC Reconf, extracts cell_group_cfg)
UEDLWorker::on_rrc_reconfig callback
    ↓
UETracker::handle_rrc_reconfig_export()
    ↓ (populates rrc_reconfig_export_t struct + packs cell_group_cfg to hex)
UETracker::on_rrc_reconfig_export callback
    ↓
Scheduler::handle_rrc_reconfig_export()
    ↓ (pushes to each InfluxWorker)
InfluxWorker → InfluxDB (measurement: "rrc_reconfig")
```

## What's in `rrc_reconfig` measurement

| Field | Type | Description |
|---|---|---|
| `rnti` | int | UE's RNTI |
| `rrc_transaction_id` | int | Transaction ID from the RRC Reconfig |
| `sp_cell_cfg_present` | bool | Whether spCellConfig is present |
| `recfg_with_sync_present` | bool | Whether reconfigurationWithSync is present (handover) |
| `phys_cell_group_cfg_present` | bool | Whether physicalCellGroupConfig is present |
| `rlc_bearer_present` | bool | Whether RLC bearers are configured |
| `mac_cell_group_cfg_present` | bool | Whether macCellGroupConfig is present |
| `pdsch_harq_ack_codebook` | int | 0=none, 1=semi-static, 2=dynamic |
| `cell_group_cfg_hex` | string | Full ASN.1-packed CellGroupConfig as hex (for reconstruction) |
| `radio_bearer_cfg_present` | bool | (future) RadioBearerConfig present |
| `ded_nas_msg_present` | bool | (future) DedicatedNAS-MessageList present |
| `meas_cfg_present` | bool | (future) MeasConfig present |
| `srb1_present` | bool | (future) SRB1 configured |
| `srb2_present` | bool | (future) SRB2 configured |
| `drb_present` | bool | (future) DRB configured |

The `radio_bearer_cfg_*` / `ded_nas_msg` / `meas_cfg` / `srb*` / `drb` fields are
set to `false` currently — they require data from the RRC Reconfig level above
cell_group_cfg (not yet captured in the callback chain).

## How to query example (InfluxQL)

```sql
-- Get all RRC Reconfig events for a specific UE
SELECT * FROM "rrc_reconfig"
WHERE "sni5gect_data_id" = 'your-data-id'
  AND "rnti" = 12345
ORDER BY time DESC

-- Check if reconfigurationWithSync (handover) is happening
SELECT "rnti", "recfg_with_sync_present", "cell_group_cfg_hex"
FROM "rrc_reconfig"
WHERE "recfg_with_sync_present" = true

-- Get the raw CellGroupConfig hex for a specific event
SELECT "cell_group_cfg_hex" FROM "rrc_reconfig"
ORDER BY time DESC LIMIT 1
```

## Limitations

- **Encryption**: The RRC Reconfig is only decoded and exported when PDCP is NOT
  encrypting (before SMC, or with NEA0). After AS security activation, the ASN.1
  unpack fails and no data is exported.
- **Per-UE only**: This is not a broadcast message like MIB/SIB1 — it's captured
  per active UE tracker. No data exists before the UE is attached.

## Files modified

- `shadower/comp/workers/influx_worker.h` — added `rrc_reconfig_export_t` struct
- `shadower/comp/workers/influx_worker.cc` — added `send_rrc_reconfig()` + dispatch
- `shadower/comp/workers/ue_dl_worker.h` — added `on_rrc_reconfig` callback
- `shadower/comp/workers/ue_dl_worker.cc` — fire callback in `handle_dlsch()`
- `shadower/comp/ue_tracker.h` — added `handle_rrc_reconfig_export()` + callback
- `shadower/comp/ue_tracker.cc` — bind callback, populate struct, fire export
- `shadower/comp/scheduler.h` — added `handle_rrc_reconfig_export()` decl
- `shadower/comp/scheduler.cc` — bind in `pre_initialize_ue()`, handler pushes to Influx
