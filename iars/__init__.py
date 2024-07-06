"""Swift IARS - Integrated Augmented Reality System, algorithm core.

Open-source release of the four algorithm blocks behind the 2024 IARS POC
prototype: redundant sensor fusion, head-motion stabilisation, binocular display
alignment and multi-sensor SLAM/INS fusion.
"""

__version__ = "1.0.0"
__all__ = ["config", "pipeline", "redundancy", "stabilization", "stereo", "slam", "sensors", "telemetry"]
