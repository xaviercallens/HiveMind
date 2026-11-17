# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase 2 & 3: SETI-Swarm Content-Addressed DHT Vault, PoUW `$HIVE` Ledger, Speculative Decoding & Swarm Fabric."""

try:
    from hivemind.swarm.decoupled_diloco import (
        DECOUPLED_DILOCO_REFERENCE,
        AdaptiveGraceWindowController,
        DecoupledDiLoCoQuorumSyncer,
        LearnerArrival,
        QuorumSelection,
        SyncerRoundTelemetry,
        direct_weighted_average,
        radial_directional_average,
        select_quorum,
        token_weighted_quality_weight,
    )
    from hivemind.swarm.diloco_nextgen import (
        AutoLoCoIntervalCorrector,
        CGADStepTelemetry,
        ChunkedOrthonormalDCTTransform,
        ComputeVisibleSparseLoCoCompressor,
        CosineGatedAdamDecayOuterOptimizer,
        CPUOffloadedFlatBucketOuterState,
        FoMoEPartialReplicaPlanner,
        HeLoCoDirectionCorrector,
        LEVERAGED_GITHUB_REPOSITORIES,
        MuonNewtonSchulzLoRAPreconditioner,
        ReputationWeightedIsoCTSMerger,
        SkipAttentionSpeculativeRouter,
        TSVDecorrelatedMerger,
    )
    from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
    from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SpeculativeSwarmResult, SwarmPeerNode
    from hivemind.swarm.ledger import (
        HivePoUWBlock,
        HiveSwarmLedger,
        InferenceSettlementReceipt,
        StorageProofReceipt,
        WorkReceipt,
    )
    from hivemind.swarm.swarm_fabric import (
        AsyncQUICGeoRingTransport,
        CompressedHiddenStatePacket,
        GeoRingPeer,
        HiveLayer2StateChannel,
        OptimisticProofOfInferenceAuditor,
        PinnedRAMExpertDMAStreamer,
        RegionalGeoRingTransport,
        StateChannelVoucher,
        TokenTreeSpeculativeVerifier,
        TwoLevelMacroMicroRouter,
        compress_hidden_states_fp8,
        decompress_hidden_states_fp8,
    )
    from hivemind.swarm.wan_fault_benchmark import (
        FaultProfile,
        FaultScenario,
        run_fault_scenario,
        run_wan_quic_turn_fault_benchmark,
        write_fault_report,
    )
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [
    "DECOUPLED_DILOCO_REFERENCE",
    "AdaptiveGraceWindowController",
    "DecoupledDiLoCoQuorumSyncer",
    "LearnerArrival",
    "QuorumSelection",
    "SyncerRoundTelemetry",
    "direct_weighted_average",
    "radial_directional_average",
    "select_quorum",
    "token_weighted_quality_weight",
    "AsyncQUICGeoRingTransport",
    "AutoLoCoIntervalCorrector",
    "CGADStepTelemetry",
    "ChunkedOrthonormalDCTTransform",
    "CompressedHiddenStatePacket",
    "ComputeVisibleSparseLoCoCompressor",
    "ContentAddressedSwarmStore",
    "CosineGatedAdamDecayOuterOptimizer",
    "CPUOffloadedFlatBucketOuterState",
    "FaultProfile",
    "FaultScenario",
    "FoMoEPartialReplicaPlanner",
    "GeoRingPeer",
    "HeLoCoDirectionCorrector",
    "HiveLayer2StateChannel",
    "HivePoUWBlock",
    "HiveSwarmLedger",
    "InferenceSettlementReceipt",
    "LEVERAGED_GITHUB_REPOSITORIES",
    "MuonNewtonSchulzLoRAPreconditioner",
    "OptimisticProofOfInferenceAuditor",
    "PinnedRAMExpertDMAStreamer",
    "RegionalDHTSwarmRing",
    "RegionalGeoRingTransport",
    "ReputationWeightedIsoCTSMerger",
    "SkipAttentionSpeculativeRouter",
    "SpeculativeSwarmResult",
    "StateChannelVoucher",
    "StorageProofReceipt",
    "SwarmPeerNode",
    "TokenTreeSpeculativeVerifier",
    "TSVDecorrelatedMerger",
    "TwoLevelMacroMicroRouter",
    "WorkReceipt",
    "compress_hidden_states_fp8",
    "decompress_hidden_states_fp8",
    "run_fault_scenario",
    "run_wan_quic_turn_fault_benchmark",
    "write_fault_report",
]


