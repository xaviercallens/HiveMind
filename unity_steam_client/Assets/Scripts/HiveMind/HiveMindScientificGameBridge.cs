using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
#if !DISABLESTEAMWORKS && STEAMWORKS_NET
using Steamworks;
#endif

namespace HiveMind.ScientificGame
{
    /// <summary>
    /// Bridges Sebastian Lague's Solar-System N-Body Physics (NBodySimulation / CelestialBody),
    /// Riley Labrecque's Steamworks.NET (AppID 3133700 local dev config), CoplayDev Unity-MCP,
    /// and the Google Antigravity + HiveMind Scientific Discovery Game backend.
    ///
    /// Strictly Non-Monetary Charter (0.00 EUR / 0.00 USD): all game points (PTS) and HMT
    /// compute-reciprocity units are educational and scientific accounting metrics only.
    /// </summary>
    public class HiveMindScientificGameBridge : MonoBehaviour
    {
        [Header("HiveMind Scientific Game Backend & OpenRouter")]
        public string localHudBaseUrl = "http://127.0.0.1:8090";
        public string openRouterBaseUrl = "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1";
        public string preferredSpecialistBee = "hivemind/bee-stars";
        public string gamerTag = "CyberValkyrie_FR";
        public string steamId64 = "76561198000000001";

        [Header("Symplectic N-Body Physics & Hamiltonian Audit")]
        public bool useSymplecticVelocityVerlet = true;
        public double initialHamiltonianEnergy = 0.0;
        public double currentHamiltonianDriftRatio = 0.0;
        public double maxAllowedHamiltonianDrift = 1e-4;

        [Header("Zero-Intrusion <1.5s GPU Preemption Telemetry")]
        public float lastPreemptionEvictionMs = 0.0f;
        public bool backgroundWorkerPausedForGame = false;

        private CelestialBody[] _bodies;
        private bool _hamiltonianInitialized = false;

        void Start()
        {
            _bodies = FindObjectsOfType<CelestialBody>();
            if (_bodies != null && _bodies.Length > 0)
            {
                initialHamiltonianEnergy = ComputeTotalHamiltonian(_bodies);
                _hamiltonianInitialized = true;
            }
            UpdateSteamRichPresence("Mission 1: Exoplanet Symplectic Transit", 0);
            StartCoroutine(NotifyForegroundGameFocusCoroutine(true));
        }

        void OnApplicationFocus(bool hasFocus)
        {
            if (gameObject.activeInHierarchy)
            {
                StartCoroutine(NotifyForegroundGameFocusCoroutine(hasFocus));
            }
        }

        /// <summary>
        /// Upgrades SebLague/Solar-System's semi-implicit Euler step to 2nd-order Symplectic
        /// Velocity-Verlet (Kick-Drift-Kick) so orbital Hamiltonian drift |Delta H / H_0| stays <= 1e-4.
        /// </summary>
        public void StepSymplecticVerlet(float dt)
        {
            if (_bodies == null || _bodies.Length == 0)
            {
                _bodies = FindObjectsOfType<CelestialBody>();
                if (_bodies == null || _bodies.Length == 0) return;
            }

            int n = _bodies.Length;
            Vector3[] accelHalf = new Vector3[n];
            for (int i = 0; i < n; i++)
            {
                accelHalf[i] = NBodySimulation.CalculateAcceleration(_bodies[i].Position, _bodies[i]);
                _bodies[i].UpdateVelocity(accelHalf[i], 0.5f * dt);
            }

            for (int i = 0; i < n; i++)
            {
                _bodies[i].UpdatePosition(dt);
            }

            for (int i = 0; i < n; i++)
            {
                Vector3 accelNew = NBodySimulation.CalculateAcceleration(_bodies[i].Position, _bodies[i]);
                _bodies[i].UpdateVelocity(accelNew, 0.5f * dt);
            }

            if (_hamiltonianInitialized && Math.Abs(initialHamiltonianEnergy) > 1e-12)
            {
                double currentH = ComputeTotalHamiltonian(_bodies);
                currentHamiltonianDriftRatio = Math.Abs((currentH - initialHamiltonianEnergy) / initialHamiltonianEnergy);
            }
        }

        /// <summary>
        /// Computes the exact N-Body mechanical Hamiltonian H = T + V across all active CelestialBody instances.
        /// </summary>
        public static double ComputeTotalHamiltonian(CelestialBody[] bodies)
        {
            double kinetic = 0.0;
            double potential = 0.0;
            int n = bodies.Length;
            for (int i = 0; i < n; i++)
            {
                double m_i = bodies[i].mass;
                double vSq = bodies[i].velocity.sqrMagnitude;
                kinetic += 0.5 * m_i * vSq;

                for (int j = i + 1; j < n; j++)
                {
                    double m_j = bodies[j].mass;
                    double dist = Vector3.Distance(bodies[i].Position, bodies[j].Position);
                    if (dist > 1e-6)
                    {
                        potential -= (Universe.gravitationalConstant * m_i * m_j) / dist;
                    }
                }
            }
            return kinetic + potential;
        }

        /// <summary>
        /// Signals the local HiveMind Zero-Intrusion Daemon to preempt background GPU training
        /// within < 1500 ms (500 ms safety margin under the 2000 ms Windows WDDM TDR timeout).
        /// </summary>
        public IEnumerator NotifyForegroundGameFocusCoroutine(bool gameActive)
        {
            string url = localHudBaseUrl.TrimEnd('/') + "/api/scientific-game/preempt";
            string payload = "{\"gamer_tag\":\"" + gamerTag + "\",\"game_exe\":\"HiveMind_Unity_SolarSystem.exe\",\"foreground_active\":" + (gameActive ? "true" : "false") + "}";
            using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(payload);
                req.uploadHandler = new UploadHandlerRaw(bodyRaw);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 2;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    backgroundWorkerPausedForGame = gameActive;
                }
            }
        }

        /// <summary>
        /// Submits a scientific mission solution (Mission 1 Exoplanet Transit, Mission 2 E8/A2 Lattice,
        /// or Mission 3 ANSE Energy-Bee Reactor) to the local HiveMind Scientific Game Engine.
        /// </summary>
        public IEnumerator SubmitScientificMissionCoroutine(string missionId, Action<string> onCompleted)
        {
            string url = localHudBaseUrl.TrimEnd('/') + "/api/scientific-game/solve";
            string payload = "{\"gamer_tag\":\"" + gamerTag + "\",\"steam_id64\":\"" + steamId64 + "\",\"mission_id\":\"" + missionId + "\"}";
            using (UnityWebRequest req = new UnityWebRequest(url, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(payload);
                req.uploadHandler = new UploadHandlerRaw(bodyRaw);
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 5;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success && onCompleted != null)
                {
                    onCompleted(req.downloadHandler.text);
                }
            }
        }

        public void UpdateSteamRichPresence(string missionName, int scientificPoints)
        {
#if !DISABLESTEAMWORKS && STEAMWORKS_NET
            if (SteamManager.Initialized)
            {
                SteamFriends.SetRichPresence("status",
                    string.Format("HiveMind Science Odyssey: {0} | {1} PTS (0 EUR Reciprocity)", missionName, scientificPoints));
                SteamFriends.SetRichPresence("steam_display", "#Status_ScientificOdyssey");
            }
#endif
        }
    }
}
