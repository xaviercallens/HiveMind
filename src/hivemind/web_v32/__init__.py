"""HiveMind web portal v3.2: one sober page, server-side sign-in against the gateway, only sourced facts."""
from hivemind.web_v32.app import GatewayClient, mount_web_v32

__all__ = ["GatewayClient", "mount_web_v32"]
