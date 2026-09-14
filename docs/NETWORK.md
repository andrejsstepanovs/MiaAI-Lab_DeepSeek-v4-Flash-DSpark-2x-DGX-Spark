# NETWORK.md — RoCE fabric setup (2x DGX Spark, TP=2)

Recipe for the two-lane ConnectX fabric used by the DSpark cluster. Record of
what was actually configured on 2026-09-14 (dual-PF A/B, commit `a6fb48c`), so
the setup can be rebuilt from scratch after a wipe.

## Hardware / topology

- Each Spark has one ConnectX-7 ASIC: 2 physical ports (p0, p1), **2 PCIe
  functions (PFs) per port**, each PF is PCIe Gen5 x4 (~126 Gb/s).
- Only **port p0 is cabled**, head-to-worker (no switch). p1 shows
  `NO-CARRIER` on both nodes.
- Both PFs of p0 sit on the **same wire** — "2 lanes" means the two PCIe
  functions feeding the one 200 Gb/s port, not two cables. No extra hardware.
- HCA names are identical on both nodes: `rocep1s0f0`, `rocep1s0f1`,
  `roceP2p1s0f0`, `roceP2p1s0f1` (`ibv_devinfo -l`). The cabled pair is
  `rocep1s0f0` (primary) + `roceP2p1s0f0` (second lane).

## Interfaces (per node)

| netdev | HCA | role | config |
|---|---|---|---|
| `enp1s0f0np0` | `rocep1s0f0` | primary lane | MTU 9000, IP `192.168.100.1` (head) / `192.168.100.2` (worker), RoCE `active_mtu 4096`, rings 8192 |
| `enP2p1s0f0np0` | `roceP2p1s0f0` | second lane | MTU 9000, IP `10.0.122.1` (head) / `10.0.122.2` (worker) — **distinct subnet**, never reuse `192.168.100.0/24` (two PFs of one port on one subnet = ARP flux), rings 8192 |

The launcher (`.env.dspark`) points NCCL at both:

```
NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0
WORKER_NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0
```

The leading `=` in the value is NCCL's exact-name selector (env assignment is
`KEY=` + value starting with `=` → `KEY==...`). NCCL needs a routable IPv4 on
every selected HCA: the second PF's own-addr GID is what the launcher's GID
validator matches on; without the IP the start script fails closed ("no usable
RoCEv2 GID"). `NCCL_CROSS_NIC=1` is set; GID indexes stay on auto
(`NCCL_IB_GID_AUTO=1`).

## Recipe — reproduce on a fresh OS install

Run on **both** nodes unless marked. NetworkManager owns these interfaces
(netplan exists but NM is the active manager for the CX7 ports).

```bash
# 1. Link settings (runtime = immediate; NM persists step 2, not this)
sudo ip link set dev enP2p1s0f0np0 mtu 9000
sudo ethtool -G enP2p1s0f0np0 rx 8192 tx 8192
sudo ethtool -G enp1s0f0np0   rx 8192 tx 8192

# 2. Static IP on the second lane via NM (the auto-created profile is called
#    "Wired connection 1" on both nodes; it otherwise DHCP-spams the link and
#    wipes manually-added addresses). Head gets .1, worker gets .2:
sudo nmcli con mod "Wired connection 1" ipv4.method manual \
  ipv4.addresses 10.0.122.1/24 ipv6.method ignore     # head
sudo nmcli con mod "Wired connection 1" ipv4.method manual \
  ipv4.addresses 10.0.122.2/24 ipv6.method ignore     # worker
sudo nmcli con up "Wired connection 1"
```

MTU 9000 + rings are **runtime-only** and do NOT survive reboot. To persist,
extend the same NM profile (`nmcli con mod "Wired connection 1" 802-3-ethernet.mtu 9000`)
or a systemd/udev unit; until then, re-run step 1 after a boot. The primary
interface's MTU 9000 is already persisted elsewhere (pre-existing config).

## Verify

```bash
# link + RoCE MTU (want active_mtu 4096 on BOTH HCAs):
ip -d link show enP2p1s0f0np0 | grep mtu
ibv_devinfo -d roceP2p1s0f0 | grep active_mtu
ping -c3 10.0.122.2            # head -> worker, second lane

# fabric bandwidth, both nodes' containers running (one rank per node):
#   scripts/nccl-fabric-bench.py — see file docstring for the docker exec pair
#   add -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET to prove device use
```

Good NCCL log lines:

```
NET/IB : Using [0]rocep1s0f0:1/RoCE [1]roceP2p1s0f0:1/RoCE [RO]; OOB enp1s0f0np0:192.168.100.1
NET/IB : Made virtual device [0] name=rocep1s0f0 speed=200000
NET/IB : Made virtual device [1] name=roceP2p1s0f0 speed=200000
```

If only one device appears, NCCL is not using the second lane — check the IP,
the selector syntax, and the GID validation output of `start-*.sh`.

## Measured effect (2026-09-14)

| all_reduce | single PF | dual PF |
|---|---|---|
| 256 KB (decode collectives) | 1.7 GB/s | 1.8 GB/s (latency-bound, unchanged as expected) |
| 8 MB (prefill chunk) | 8.6 GB/s | 10.1 GB/s (+18%) |
| 64 MB | 10.4 GB/s | 21.4 GB/s (+105%) |

vLLM end-to-end: no measurable change at 7.5k/60k prompts (prefill comm is a
small share of a chunk; decode is latency-bound). Kept because it costs
nothing and removes the PCIe bottleneck for large transfers. Note: GPUDirect
RDMA is unavailable on GB10 (`DMA_BUF_SUPPORTED=0`) — every collective bounces
through host memory; this is expected and unrelated to the lane count.

## Rollback

```bash
# revert to single lane in .env.dspark:
#   NCCL_IB_HCA==rocep1s0f0   /   WORKER_NCCL_IB_HCA==rocep1s0f0
sudo ip addr del 10.0.122.1/24 dev enP2p1s0f0np0   # + restart the stack
```
