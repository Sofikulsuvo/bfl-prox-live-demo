import io
import os
import base64
import hashlib
import time
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from web3 import Web3
import solcx
from sklearn.metrics import f1_score, accuracy_score

st.set_page_config(page_title="BFL-Prox Live Cloud Simulator", layout="wide")

CLASS_NAMES = ["Adware", "Banking", "SMS", "Riskware", "Benign"]
NUM_CLASSES = 5
IN_DIM = 470

# ==========================================================
# 1. SOLIDITY SMART CONTRACT (BFLReputation.sol)
# ==========================================================
SOLIDITY_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BFLReputation {
    struct Client {
        uint8 reputation;
        bool isRegistered;
        string latestCid;
    }

    address public validatorCommittee;
    uint256 public currentRound = 1;
    string public globalModelCid;
    mapping(address => Client) public clients;

    event UpdateSubmitted(uint256 indexed round, address indexed client, string cid);
    event ReputationUpdated(address indexed client, uint8 newReputation, bool quarantined);
    event GlobalModelCommitted(uint256 indexed round, string globalCid);

    constructor() {
        validatorCommittee = msg.sender;
    }

    function registerClient(address _client) external {
        require(msg.sender == validatorCommittee, "Only committee");
        clients[_client] = Client(100, true, "");
    }

    function submitUpdate(string calldata _cid) external {
        require(clients[msg.sender].isRegistered, "Unregistered device");
        require(clients[msg.sender].reputation >= 30, "Quarantined: reputation < 30");
        clients[msg.sender].latestCid = _cid;
        emit UpdateSubmitted(currentRound, msg.sender, _cid);
    }

    function slashOrReward(address _client, bool _isByzantine) external {
        require(msg.sender == validatorCommittee, "Only committee");
        uint8 rep = clients[_client].reputation;
        if (_isByzantine) {
            rep = rep >= 40 ? rep - 40 : 0;
        } else {
            rep = rep <= 95 ? rep + 5 : 100;
        }
        clients[_client].reputation = rep;
        emit ReputationUpdated(_client, rep, rep < 30);
    }

    function commitGlobalModel(string calldata _globalCid) external {
        require(msg.sender == validatorCommittee, "Only committee");
        globalModelCid = _globalCid;
        currentRound += 1;
        emit GlobalModelCommitted(currentRound - 1, _globalCid);
    }

    function getReputation(address _client) external view returns (uint8) {
        return clients[_client].reputation;
    }
}
"""

# ==========================================================
# 2. PROPOSED MODEL: LightweightMalwareMLP (154,117 Params)
# ==========================================================
class LightweightMalwareMLP(nn.Module):
    def __init__(self, in_dim=470, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def pin_to_offchain_ipfs(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw_bytes = buf.getvalue()
    size_kb = round(len(raw_bytes) / 1024.0, 2)
    sha256_digest = hashlib.sha256(raw_bytes).digest()
    cid_bytes = b"\x01\x55\x12\x20" + sha256_digest
    cid_b32 = "b" + base64.b32encode(cid_bytes).decode("ascii").lower().rstrip("=")
    return cid_b32, size_kb, state_dict


# ==========================================================
# 3. EMBEDDED EVM SMART CONTRACT INITIALIZATION
# ==========================================================
@st.cache_resource
def init_embedded_evm():
    solc_dir = "/tmp/solcx"
    os.makedirs(solc_dir, exist_ok=True)
    solc_bin = os.path.join(solc_dir, "solc-v0.8.20")

    if not os.path.exists(solc_bin):
        url = (
            "https://raw.githubusercontent.com/ethereum/solc-bin/gh-pages/"
            "linux-amd64/solc-linux-amd64-v0.8.20%2Bcommit.a1b79de6"
        )
        urllib.request.urlretrieve(url, solc_bin)
        os.chmod(solc_bin, 0o755)

    solcx.set_solc_version("0.8.20", solcx_binary_path=solc_dir)

    compiled = solcx.compile_source(
        SOLIDITY_SOURCE,
        output_values=["abi", "bin"],
        solcx_binary_path=solc_dir
    )
    _, contract_interface = compiled.popitem()
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]

    w3 = Web3(Web3.EthereumTesterProvider())
    committee_addr = w3.eth.accounts[0]
    phone_addrs = [w3.eth.accounts[1], w3.eth.accounts[2], w3.eth.accounts[3]]

    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = Contract.constructor().transact({"from": committee_addr})
    rcpt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract = w3.eth.contract(address=rcpt.contractAddress, abi=abi)

    for addr in phone_addrs:
        reg_tx = contract.functions.registerClient(addr).transact({"from": committee_addr})
        w3.eth.wait_for_transaction_receipt(reg_tx)

    return w3, contract, committee_addr, phone_addrs


# ==========================================================
# 4. NON-IID 3-PHONE DATASET GENERATOR
# ==========================================================
@st.cache_resource
def load_3phone_datasets():
    rng = np.random.default_rng(42)
    centroids = rng.uniform(0.05, 0.45, size=(NUM_CLASSES, IN_DIM)).astype(np.float32)
    for c in range(NUM_CLASSES):
        centroids[c, c * 80 : (c + 1) * 80] += 0.45

    X_tr_list, y_tr_list, X_te_list, y_te_list = [], [], [], []
    for c in range(NUM_CLASSES):
        Xc = np.clip(rng.normal(centroids[c], 0.22, size=(400, IN_DIM)), 0.0, 1.0).astype(np.float32)
        X_tr_list.append(Xc[:320])
        y_tr_list.append(np.full(320, c))
        X_te_list.append(Xc[320:])
        y_te_list.append(np.full(80, c))

    X_tr, y_tr = np.vstack(X_tr_list), np.concatenate(y_tr_list)
    X_te, y_te = np.vstack(X_te_list), np.concatenate(y_te_list)

    def sample_mix(mix_dict, seed):
        r = np.random.default_rng(seed)
        Xs, ys = [], []
        for cls_id, count in mix_dict.items():
            idx = np.where(y_tr == cls_id)[0]
            chosen = r.choice(idx, min(count, len(idx)), replace=False)
            Xs.append(X_tr[chosen])
            ys.append(y_tr[chosen])
        return (
            torch.tensor(np.vstack(Xs), dtype=torch.float32),
            torch.tensor(np.concatenate(ys), dtype=torch.long),
        )

    phones = [
        ("Phone 1 (Honest - Banking/Benign)", sample_mix({4: 180, 1: 50}, seed=101)),
        ("Phone 2 (Honest - SMS/Adware)", sample_mix({2: 160, 0: 70}, seed=102)),
        ("Phone 3 (Target - Banking/Riskware)", sample_mix({1: 140, 3: 90}, seed=103)),
    ]
    return phones, torch.tensor(X_te, dtype=torch.float32), y_te


w3, contract, committee_addr, phone_addrs = init_embedded_evm()
phones_data, X_test_t, y_test_np = load_3phone_datasets()

if "bootstrapped" not in st.session_state:
    torch.manual_seed(42)
    st.session_state.global_model = LightweightMalwareMLP(IN_DIM, NUM_CLASSES)
    st.session_state.ipfs_store = {}

    init_cid, init_kb, init_sd = pin_to_offchain_ipfs(st.session_state.global_model.state_dict())
    st.session_state.ipfs_store[init_cid] = init_sd

    tx = contract.functions.commitGlobalModel(init_cid).transact({"from": committee_addr})
    rcpt = w3.eth.wait_for_transaction_receipt(tx)

    st.session_state.chain_txs = [
        {
            "Block": rcpt.blockNumber,
            "Event": "GenesisGlobalModel",
            "Sender": committee_addr[:10] + "...",
            "IPFS_CID": init_cid[:22] + "...",
            "Payload_KB": init_kb,
            "Tx_Hash": rcpt.transactionHash.hex()[:18] + "...",
            "Gas_Used": rcpt.gasUsed,
        }
    ]
    st.session_state.audit_logs = []
    st.session_state.f1_history = []
    st.session_state.bootstrapped = True


# ==========================================================
# 5. FEDERATED ROUND EXECUTION (FedProx + IPFS + EVM TXS)
# ==========================================================
def run_live_round(phone3_mode, mu=0.05):
    onchain_round = contract.functions.currentRound().call() - 1
    global_model = st.session_state.global_model
    global_sd = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    global_params = [p.detach().clone() for p in global_model.parameters()]
    criterion = nn.CrossEntropyLoss()

    submitted_cids = []

    for idx, (name, (Xc, yc)) in enumerate(phones_data):
        addr = phone_addrs[idx]
        rep = contract.functions.getReputation(addr).call()

        if rep < 30:
            st.session_state.audit_logs.append(
                {
                    "Round": onchain_round,
                    "Device": name,
                    "Wallet": addr[:10] + "...",
                    "Status": "QUARANTINED (Contract Blocked)",
                    "IPFS_CID": "-",
                    "Cosine": "-",
                    "Norm_Ratio": "-",
                    "Weight_Alpha": 0.0,
                    "OnChain_Rep": rep,
                }
            )
            continue

        yc_train = yc.clone()
        if idx == 2 and phone3_mode == "Label-Flipping (Malware -> Benign)":
            yc_train = torch.where(yc_train != 4, torch.tensor(4), yc_train)

        local_model = LightweightMalwareMLP(IN_DIM, NUM_CLASSES)
        local_model.load_state_dict(global_sd)
        opt = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        local_model.train()

        for _ in range(3):
            perm = torch.randperm(len(Xc))
            for i in range(0, len(Xc), 32):
                b = perm[i : i + 32]
                opt.zero_grad()
                loss = criterion(local_model(Xc[b]), yc_train[b])
                prox = sum(
                    ((lw - gw) ** 2).sum()
                    for lw, gw in zip(local_model.parameters(), global_params)
                )
                loss += (mu / 2.0) * prox
                loss.backward()
                opt.step()

        local_sd = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
        if idx == 2 and phone3_mode == "Sign-Flipping (-2x Update)":
            for k in local_sd:
                local_sd[k] = global_sd[k] - 2.0 * (local_sd[k] - global_sd[k])

        cid, size_kb, stored_sd = pin_to_offchain_ipfs(local_sd)
        st.session_state.ipfs_store[cid] = stored_sd

        tx = contract.functions.submitUpdate(cid).transact({"from": addr})
        rcpt = w3.eth.wait_for_transaction_receipt(tx)
        st.session_state.chain_txs.append(
            {
                "Block": rcpt.blockNumber,
                "Event": f"UpdateSubmitted ({name.split(' (')[0]})",
                "Sender": addr[:10] + "...",
                "IPFS_CID": cid[:22] + "...",
                "Payload_KB": size_kb,
                "Tx_Hash": rcpt.transactionHash.hex()[:18] + "...",
                "Gas_Used": rcpt.gasUsed,
            }
        )
        submitted_cids.append((name, addr, cid, len(Xc)))

    pulled_updates = [
        (name, addr, st.session_state.ipfs_store[cid], cid, n_s)
        for name, addr, cid, n_s in submitted_cids
    ]

    head_deltas = torch.stack(
        [
            (sd["net.6.weight"] - global_sd["net.6.weight"]).flatten()
            for _, _, sd, _, _ in pulled_updates
        ]
    )
    med_head = torch.median(head_deltas, dim=0).values
    med_head_norm = torch.norm(med_head) + 1e-8

    full_norms = torch.tensor(
        [
            torch.norm(torch.cat([(sd[k] - global_sd[k]).flatten() for k in global_sd]))
            for _, _, sd, _, _ in pulled_updates
        ]
    )
    med_norm = torch.median(full_norms) + 1e-8

    weights_alpha, clipped_sds = [], []
    for i, (name, addr, sd, cid, n_samples) in enumerate(pulled_updates):
        hd = head_deltas[i]
        cos_sim = float(torch.dot(hd, med_head) / ((torch.norm(hd) + 1e-8) * med_head_norm))
        norm_ratio = float(full_norms[i] / med_norm)
        is_byzantine = (cos_sim < -0.20) or (norm_ratio > 2.2)

        slash_tx = contract.functions.slashOrReward(addr, is_byzantine).transact(
            {"from": committee_addr}
        )
        w3.eth.wait_for_transaction_receipt(slash_tx)
        new_rep = contract.functions.getReputation(addr).call()

        clip_scale = min(1.0, float(med_norm / (full_norms[i] + 1e-8)))
        clipped_sds.append(
            {k: global_sd[k] + clip_scale * (sd[k] - global_sd[k]) for k in global_sd}
        )

        if new_rep >= 30 and not is_byzantine:
            alpha = (new_rep / 100.0) * max(0.10, cos_sim) * n_samples
        else:
            alpha = 0.0
        weights_alpha.append(alpha)

        st.session_state.audit_logs.append(
            {
                "Round": onchain_round,
                "Device": name,
                "Wallet": addr[:10] + "...",
                "Status": "SLASHED (-40)" if is_byzantine else "VERIFIED (+5)",
                "IPFS_CID": cid[:22] + "...",
                "Cosine": round(cos_sim, 4),
                "Norm_Ratio": round(norm_ratio, 2),
                "Weight_Alpha": round(alpha, 2),
                "OnChain_Rep": new_rep,
            }
        )

    tot_alpha = sum(weights_alpha)
    if tot_alpha > 0:
        new_global_sd = {
            k: sum((weights_alpha[i] / tot_alpha) * clipped_sds[i][k] for i in range(len(clipped_sds)))
            for k in global_sd
        }
        global_model.load_state_dict(new_global_sd)

    global_model.eval()
    with torch.no_grad():
        preds = torch.argmax(global_model(X_test_t), dim=1).numpy()
        macro_f1 = float(f1_score(y_test_np, preds, average="macro"))
        acc = float(accuracy_score(y_test_np, preds) * 100.0)

    g_cid, g_kb, g_stored_sd = pin_to_offchain_ipfs(global_model.state_dict())
    st.session_state.ipfs_store[g_cid] = g_stored_sd
    commit_tx = contract.functions.commitGlobalModel(g_cid).transact({"from": committee_addr})
    commit_rcpt = w3.eth.wait_for_transaction_receipt(commit_tx)

    st.session_state.chain_txs.append(
        {
            "Block": commit_rcpt.blockNumber,
            "Event": f"GlobalModelCommitted (R{onchain_round})",
            "Sender": committee_addr[:10] + "...",
            "IPFS_CID": g_cid[:22] + "...",
            "Payload_KB": g_kb,
            "Tx_Hash": commit_rcpt.transactionHash.hex()[:18] + "...",
            "Gas_Used": commit_rcpt.gasUsed,
        }
    )
    st.session_state.f1_history.append(
        {"Round": onchain_round, "Macro_F1": macro_f1, "Accuracy_%": acc}
    )


# ==========================================================
# 6. STREAMLIT LIVE CLOUD UI
# ==========================================================
st.title("BFL-Prox: Live Online 3-Phone Blockchain-FL Simulator")
st.caption(
    f"Live Solidity Contract `{contract.address}` (EVM) + Off-Chain IPFS CIDv1 + "
    "`LightweightMalwareMLP` (602.02 KB)"
)

with st.sidebar:
    st.header("Simulation Controls")
    p3_mode = st.selectbox(
        "Phone 3 Behavior Mode:",
        [
            "Honest (Normal FedProx)",
            "Sign-Flipping (-2x Update)",
            "Label-Flipping (Malware -> Benign)",
        ],
    )
    if st.button("Execute Live Round (IPFS + EVM)", type="primary", use_container_width=True):
        run_live_round(p3_mode)
    if st.button("Execute 3 Consecutive Rounds", use_container_width=True):
        for _ in range(3):
            run_live_round(p3_mode)

cols = st.columns(3)
for idx, (name, _) in enumerate(phones_data):
    addr = phone_addrs[idx]
    rep = contract.functions.getReputation(addr).call()
    status = "ACTIVE" if rep >= 30 else "QUARANTINED (R < 30)"
    with cols[idx]:
        st.subheader(name.split(" (")[0])
        st.caption(f"Wallet: `{addr[:14]}...`")
        st.metric("On-Chain Reputation (R)", f"{rep} / 100", status)
        st.progress(rep / 100.0)

st.divider()
left_col, right_col = st.columns([1, 1.3])

with left_col:
    st.subheader("On-Device Real-Time APK Scanner")
    sample_idx = st.slider("Test APK Sample Index", 0, len(X_test_t) - 1, 0)
    sample_x = X_test_t[sample_idx : sample_idx + 1]
    true_label = CLASS_NAMES[int(y_test_np[sample_idx])]

    st.session_state.global_model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        probs = torch.softmax(st.session_state.global_model(sample_x), dim=1).numpy()[0]
        lat_ms = (time.perf_counter() - t0) * 1000.0

    pred_label = CLASS_NAMES[int(np.argmax(probs))]
    st.write(
        f"**True:** `{true_label}` | **Predicted:** `{pred_label}` | **Latency:** `{lat_ms:.3f} ms`"
    )
    st.bar_chart(pd.DataFrame({"Category": CLASS_NAMES, "Confidence": probs}).set_index("Category"))

with right_col:
    st.subheader("Global Model Macro-F1 Progression")
    if st.session_state.f1_history:
        st.line_chart(pd.DataFrame(st.session_state.f1_history).set_index("Round")["Macro_F1"])
    else:
        st.info("Click 'Execute Live Round (IPFS + EVM)' in the sidebar to begin.")

st.divider()
st.subheader("Validator Committee Cosine & Reputation Audit")
if st.session_state.audit_logs:
    st.dataframe(
        pd.DataFrame(st.session_state.audit_logs).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Live EVM Smart Contract Transaction & IPFS CID Explorer")
st.dataframe(
    pd.DataFrame(st.session_state.chain_txs).iloc[::-1],
    use_container_width=True,
    hide_index=True,
)
