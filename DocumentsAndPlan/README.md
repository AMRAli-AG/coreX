
<div align="center">
<img width="288" height="128" alt="2-removebg-preview" src="https://github.com/user-attachments/assets/e49f93a0-99ac-4b9c-981c-5ea56bf3e214" />

![CoreX Logo](https://img.shields.io/badge/CoreX-Predictive%20Maintenance-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge)
![Phase](https://img.shields.io/badge/Phase-Research%20%26%20Planning-green?style=for-the-badge)

**AI-Driven Digital Twin Framework for Predictive Maintenance and Operational Optimization in Industrial Robotic Systems**

</div>

---

## 📋 Project Overview

CoreX combines **Artificial Intelligence** and **Digital Twin** technologies to enable predictive maintenance and operational optimization for arm robotic systems.

### 🎯 Key Objectives

- **Model Development**: Digital twin with ≥95% accuracy
- **Predictive Intelligence**: AI models with ≥90% accuracy
- **Operational Insight**: 30% reduction in unplanned downtime
- **Validation Framework**: ≤5% variance from physical system

---

## ✨ Key Features

- Real-time anomaly detection and fault classification
- Remaining Useful Life (RUL) prediction
- High-fidelity Digital Twin simulation
- End-to-end security (AES/TLS, SIEM)
- Live monitoring dashboard (ThingsBoard)
- Edge deployment (Jetson Nano / Raspberry Pi)

---

## 🏗 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Physical Layer                           │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │  KUKA Robot  │────────▶│  Black Box      │              │
│  │              │ Ethernet │  (MCU Module)   │              │
│  └──────────────┘         └─────────────────┘              │
└────────────────────────────────┬────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────┐
│                   Edge Computing Layer                       │
│  ┌────────────────┐    ┌──────────────────┐                │
│  │  Data          │───▶│  AI Models       │                │
│  │  Acquisition   │    │  - Anomaly Det.  │                │
│  └────────────────┘    │  - RUL Pred.     │                │
│                        │  - Fault Class.  │                │
│                        └──────────────────┘                │
└────────────────────────────────┬────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────┐
│                   Software Layer                             │
│  ┌────────────────┐    ┌──────────────────┐                │
│  │  Digital Twin  │◀──▶│  Dashboard       │                │
│  │  Simulation    │    │  (ThingsBoard)   │                │
│  └────────────────┘    └──────────────────┘                │
│                                                              │
│  ┌────────────────────────────────────────┐                │
│  │         Security Layer (SIEM)          │                │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

---

## 📅 Complete Project Plan

**Duration:** 9.5 months (Oct 2025 - July 2026) | **Teams:** Embedded, AI, Security

### Phase Summary

| Phase | Name | Duration | Period | Teams Involved |
|-------|------|----------|--------|----------------|
| **Phase 1** | Research & Planning | 2 months | Oct 1 - Dec 2, 2025 | All Teams |
| **Phase 2** | Design & Prototyping | 4.5 months | Dec 3, 2025 - Apr 14, 2026 | All Teams (Parallel) |
| **Phase 3** | Integration & Dashboard | 1.5 months | Apr 15 - May 26, 2026 | All Teams (Parallel) |
| **Phase 4** | Testing & Validation | 2 months | May 27 - July 21, 2026 | All Teams |

---
### Phase 1: Research & Planning (Oct 1 - Dec 2, 2025)

| Milestone | Duration | Start Date | End Date |
|-----------|----------|------------|----------|
| Literature Review | 2 weeks | Oct 1, 2025 | Oct 14, 2025 |
| Requirement Analysis | 2 weeks | Oct 15, 2025 | Oct 28, 2025 |
| System Architecture & Planning | 3 weeks | Oct 29, 2025 | Nov 18, 2025 |
| Component Selection & Sourcing | 2 weeks | Nov 19, 2025 | Dec 2, 2025 |

---

### Phase 2: Design & Prototyping (Dec 3, 2025 - Apr 14, 2026)

| Milestone | Team | Duration | Start Date | End Date | Output |
|-----------|------|----------|------------|----------|--------|
| **Dec 3 - Dec 16, 2025** ||||||
| Data Acquisition from KUKA Robot | Embedded | 2 weeks | Dec 3, 2025 | Dec 16, 2025 | Robot telemetry dataset |
| Problem Definition & Data Requirements | AI | 2 weeks | Dec 3, 2025 | Dec 16, 2025 | Problem statement |
| Port Security Hardening | Security | 2 weeks | Dec 3, 2025 | Dec 16, 2025 | Secured ports |
| **Dec 17, 2025 - Jan 6, 2026** ||||||
| Low-Resource Ubuntu VM Setup | Embedded | 2 weeks | Dec 17, 2025 | Dec 30, 2025 | Test environment |
| Data Preprocessing & Features | AI | 3 weeks | Dec 17, 2025 | Jan 6, 2026 | Clean dataset |
| Data Encryption on Ethernet | Security | 3 weeks | Dec 17, 2025 | Jan 6, 2026 | Encrypted packets |
| **Dec 31, 2025 - Jan 27, 2026** ||||||
| Run AI Model on Low-Resource VM | Embedded | 3 weeks | Dec 31, 2025 | Jan 20, 2026 | Model inference |
| Anomaly Detection Model | AI | 3 weeks | Jan 7, 2026 | Jan 27, 2026 | Trained model |
| SIEM Training + SW Security | Security | 3 weeks | Jan 7, 2026 | Jan 27, 2026 | Security monitoring skills |
| **Jan 21 - Feb 10, 2026** ||||||
| Deploy Model on Jetson/Pi | Embedded | 3 weeks | Jan 21, 2026 | Feb 10, 2026 | Hardware benchmarks |
| Fault Identification (Optional) | AI | 2 weeks | Jan 28, 2026 | Feb 10, 2026 | Fault classifier |
| SIEM Implementation | Security | 2 weeks | Jan 28, 2026 | Feb 10, 2026 | SIEM dashboard |
| **Feb 11 - Mar 3, 2026** ||||||
| ThingsBoard Dashboard | Embedded | 3 weeks | Feb 11, 2026 | Mar 3, 2026 | Live robot dashboard |
| RUL Model Development | AI | 3 weeks | Feb 11, 2026 | Mar 3, 2026 | RUL model |
| **Mar 4 - Mar 17, 2026** ||||||
| Auto-Boot AI Service | Embedded | 2 weeks | Mar 4, 2026 | Mar 17, 2026 | Auto-run AI at startup |
| Model Evaluation & Selection | AI | 2 weeks | Mar 4, 2026 | Mar 17, 2026 | Best model |
| **Mar 18 - Mar 31, 2026** ||||||
| Optimization (Energy/Stress) | AI | 2 weeks | Mar 18, 2026 | Mar 31, 2026 | Optimized trajectories |
| **Apr 1 - Apr 14, 2026** ||||||
| AI/DT Integration | AI | 2 weeks | Apr 1, 2026 | Apr 14, 2026 | Integrated AI system |

---

### Phase 3: Integration & Dashboard (Apr 15 - May 26, 2026)

| Milestone | Team | Duration | Start Date | End Date | Output |
|-----------|------|----------|------------|----------|--------|
| **Apr 15 - Apr 28, 2026** ||||||
| Connect Edge Devices to AI System | Embedded | 2 weeks | Apr 15, 2026 | Apr 28, 2026 | Real-time inference |
| AI–Dashboard Connection | AI | 2 weeks | Apr 15, 2026 | Apr 28, 2026 | Live anomaly + RUL |
| Encryption & Access Control Review | Security | 2 weeks | Apr 15, 2026 | Apr 28, 2026 | Secured robot network |
| **Apr 29 - May 12, 2026** ||||||
| Telemetry Streaming (MQTT Loop) | Embedded | 2 weeks | Apr 29, 2026 | May 12, 2026 | Data stream |
| Digital Twin Sync with Real Robot | AI | 2 weeks | Apr 29, 2026 | May 12, 2026 | Synced DT |
| SIEM Dashboard Finalization | Security | 2 weeks | Apr 29, 2026 | May 12, 2026 | Real-time alerts |
| **May 13 - May 26, 2026** ||||||
| Full Embedded Integration Testing | Embedded | 2 weeks | May 13, 2026 | May 26, 2026 | Stable pipeline |
| Full AI Pipeline Validation | AI | 2 weeks | May 13, 2026 | May 26, 2026 | Production-ready AI |
| Full Security Validation / Pen-Test | Security | 2 weeks | May 13, 2026 | May 26, 2026 | Final security report |

---

### Phase 4: Testing & Validation (May 27 - July 21, 2026)

| Milestone | Duration | Start Date | End Date |
|-----------|----------|------------|----------|
| Controlled Testing | 2 weeks | May 27, 2026 | June 9, 2026 |
| Performance Optimization | 2 weeks | June 10, 2026 | June 23, 2026 |
| Real-World Simulation | 2 weeks | June 24, 2026 | July 7, 2026 |
| Final Report & Presentation | 2 weeks | July 8, 2026 | July 21, 2026 |

---








## 📧 Contact

**CoreX Team**  
linkein [corex-project.com](https://www.linkedin.com/feed/update/urn:li:activity:7371933305350479872/)

---

## 📈 Project Status

![Progress](https://img.shields.io/badge/Progress-15%25-yellow?style=for-the-badge)
![Phase](https://img.shields.io/badge/Current%20Phase-Phase%201-green?style=for-the-badge)

**Last Updated:** November 28, 2025

---

<div align="center">

**CoreX © 2025** | Building the Future of Industrial Automation


</div>
