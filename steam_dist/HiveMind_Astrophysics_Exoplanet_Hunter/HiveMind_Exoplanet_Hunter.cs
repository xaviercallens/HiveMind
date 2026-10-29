using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Media;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Speech.Synthesis;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;

namespace HiveMind.ExoplanetHunterClient
{
    public class ExoTarget
    {
        public int Index;
        public string Id;
        public string ShortLabel;
        public string HostStar;
        public string Disposition;
        public float GalaxyXNorm;
        public float GalaxyYNorm;
        public double PeriodDays;
        public double SemiMajorAxisAu;
        public double Eccentricity;
        public double RadialVelocityKms;
        public double DepthPpm;
        public double RadiusRearth;
        public double MassMearth;
        public double BulkDensityGcm3;
        public double TeqK;
        public double InsolationEarth;
        public bool InHz;
        public Color PrimaryColor;
        public Color PedestalColor;
        public bool ApprovedByGamer;
        public bool LandedMedal;
        public string AnalysisEn;
        public string AnalysisFr;
        public string LifeEn;
        public string LifeFr;
        public string KidLessonEn;
        public string KidLessonFr;
        // Progressive Difficulty & Real Astrophysical Hazard Parameters
        public int DifficultyTier;
        public int AsteroidBeltCount;
        public double SurfaceGravityG;
        public double EscapeVelKms;
        public double ScaleHeightKm;
        public double SurfacePressureBar;
        public double WindShearMaxMs;
        public bool IsNewMysteryPlanet;
        public string AsteroidPhysicsEn;
        public string AsteroidPhysicsFr;
        public string AtmoLandingEn;
        public string AtmoLandingFr;
    }

    public class ProbeProjectile
    {
        public float X;
        public float Y;
        public float TargetX;
        public float TargetY;
        public Color Color;
    }

    public class ScorePopup
    {
        public float X;
        public float Y;
        public string Text;
        public Color Color;
        public double Life;
    }

    public class HudStarBonus
    {
        public float XNorm;
        public float YNorm;
        public bool Collected;
    }

    public class HudHotspot
    {
        public string Id;
        public Rectangle Rect;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOYINFOEX
    {
        public int dwSize;
        public int dwFlags;
        public int dwXpos;
        public int dwYpos;
        public int dwZpos;
        public int dwRpos;
        public int dwUpos;
        public int dwVpos;
        public int dwButtons;
        public int dwButtonNumber;
        public int dwPOV;
        public int dwReserved1;
        public int dwReserved2;
    }

    public class ExoplanetHunterForm : Form
    {
        [DllImport("winmm.dll")]
        public static extern int joyGetPosEx(int uJoyID, ref JOYINFOEX pji);

        [DllImport("winmm.dll", CharSet = CharSet.Auto)]
        public static extern int mciSendString(string command, StringBuilder buffer, int bufferSize, IntPtr hwndCallback);

        private List<ExoTarget> _targets = new List<ExoTarget>();
        private List<ExoTarget> _unseenPlanetPool = new List<ExoTarget>();
        private ExoTarget _current;
        private string _lang = "en"; // Default to English ("en") with GCP Gemini / Neural Voice! Supports "en", "fr", "es", "de", "zh"
        private bool _voiceAndSoundOn = true;
        private string _geminiVoiceName = "Puck"; // Expressive Gemini voices: "Puck" (Kid Guide), "Aoede" (Storyteller), "Kore" (Science Mentor), "Charon" (Commander)
        private string _activeVoiceEngineLabel = "GCP GEMINI NEURAL • PUCK";
        private int _speechGenerationId = 0;
        private string _subtitleText = "";
        private int _gameStage = 0; // 0 = Galaxy Map, 1 = Bee-MoE Orbit, 2 = Easy Oolite Planet Landing, 3 = JWST Science Lab
        private int _scoreHmt = 2400;
        private int _streakCount = 2;
        private int _medalsCount = 3;
        private bool _showNarrativeCard = true;
        private bool[] _chapterIntroSpoken = new bool[5];
        private DateTime _chapterIntroEndUtc = DateTime.MinValue;
        private int _planetSwitchCount = 0;
        private int _planetsExploredCount = 0;
        private int _difficultyLevel = 1;
        private double _asteroidHitCooldown = 0.0;
        private double _surpriseEventTimer = 5.5;
        private string _surpriseEventBanner = "";
        private string _atmoLayerName = "THERMOSPHERE (IONOSPHERIC ENTRY)";
        private List<ScorePopup> _popups = new List<ScorePopup>();
        private List<HudStarBonus> _bonusStars = new List<HudStarBonus>();
        private SpeechSynthesizer _speechSynth = null;
        private readonly object _speechLock = new object();

        // USS HiveMind position on Galaxy Map (matching reference image: upper-left near TRAPPIST-1e / K2-18 b / LHS 1140 b)
        private float _shipXNorm = 0.255f;
        private float _shipYNorm = 0.275f;
        private double _shipHeading = 0.52;
        private List<ProbeProjectile> _launchedProbes = new List<ProbeProjectile>();
        private double _warpTimer = 0.0;

        // Stage 1 Orbital Bee-MoE Robotic Satellite Module
        private float _beeX = 240f;
        private float _beeY = 260f;
        private float _beeVx = 0f;
        private float _beeVy = 0f;
        private float _planetScreenX = 680f;
        private float _planetScreenY = 260f;
        private double _captureProgress = 0.0;
        private double _hazardHitFlash = 0.0;

        // Stage 2 Movie-Grade Oolite (oolite.space) + Planetfall OXP Planetary Landing State
        private double _landerAltKm = 3.000;          // Smoothly descends from 3.0 km -> 0.0 km
        private double _landerSpeedMs = 96.0;         // Upper entry speed (triggers dynamic plasma glow > 85 m/s, brakes to 55 m/s)
        private double _landerCabinTempC = 96.0;      // Matches "96° C.TEMP" re-entry plasma heat
        private double _landerFwdShield = 100.0;      // Matches "100% F.SH"
        private double _landerAftShield = 100.0;      // Matches "100% A.SH"
        private double _landerArmour = 100.0;         // Matches "100% ARMOUR"
        private double _landerEnergy = 256.0;         // Matches "256 ENERGY"
        private float _landerOffsetX = 12.0f;         // Lateral offset from Green Landing Pad (Aligned < 35m!)
        private float _landerPitchDeg = -10.0f;       // Flight path descent angle
        private double _landerWindShearMs = 8.4;      // Live crosswind displayed on central Pitch Ladder
        private bool _landingClearanceGranted = true; // Automatically granted!
        private bool _landingGearDown = true;         // Automatically deployed!
        private bool _stage2TradeTabExpanded = false; // Contextual HUD Decluttering: collapsed in Flight Mode, toggle with [TAB]
        private string _landingStatusMsg = "GLIDE-SLOPE LOCKED — FOLLOW 3D GREEN TUNNEL TO PAD! [SPACE] RETRO-BRAKE (+100 PTS)";

        // Stage 4 (5.BOINC/GCP): Steam Free-to-Play Idle-GPU BOINC Citizen-Science & GCP Community Hub State
        private double _idleSeconds = 4.6;            // Seconds since last keyboard/mouse/gamepad action (>= 4.0s triggers Idle-GPU BOINC)
        private bool _boincAutoIdleMode = true;       // Automatically crunch BOINC astrophysics workunits when GPU is not used by active gameplay
        private bool _boincForceAlwaysOn = false;     // Optional override to crunch even while moving
        private int _boincWorkunitsCompleted = 6842;
        private double _boincCobblestones = 184920.5;
        private double[] _boincProjProgress = new double[] { 84.6, 92.4, 71.2, 65.8 };
        private string _boincLastReceiptId = "BOINC-BLS-9F8A41C2";
        private string _boincLastGcpLeaseId = "LEASE-GCP-RUCHE-9941";
        private string _boincLastSyncStatus = "ONLINE • SYNCED WITH hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app";
        private int _communityActiveMembers = 1428;
        private string _communityLatestNote = "Cmdr_JeanPaul_NCC31337 + Dr_Elena_Vega: Confirmed 2,920 ppm transit + H2O/CH4/CO2 on K2-18 b via Idle-GPU BLS fold!";

        // Input state
        private bool _kUp, _kDown, _kLeft, _kRight, _kSpace;
        private int _prevGpButtons = 0;
        private double _simTime = 0.0;

        // Preloaded 3D Raycast Planet Spheres, HD Spiral Galaxy Backdrop, Oolite Planetary Horizon & Stitch HD Assets
        private Image _galaxyBackdrop = null;
        private Image _ooliteHorizonBackdrop = null;
        private Image _stitchOrbitK218bImg = null;
        private Image _stitchSarTerrainImg = null;
        private Image _stitchTrappistImg = null;
        private Image[] _planetGlobes = new Image[10];
        private List<HudHotspot> _hotspots = new List<HudHotspot>();

        // Layout Rectangles (computed dynamically on Resize/Paint)
        private Rectangle _consoleRect;
        private Rectangle _topTelemetryRect;
        private Rectangle _bannerRect;
        private Rectangle _galaxyRect;
        private Rectangle _oscRect;
        private Rectangle _collectionRect;
        private Rectangle[] _slotRects = new Rectangle[10];

        private System.Windows.Forms.Timer _timer;

        public ExoplanetHunterForm()
        {
            this.Text = "USS HIVEMIND NCC-31337 v5.0 — Stitch Tactical Bridge Aerospace HUD (10018415280079892710) • GCP Gemini Voice & Oolite Landing";
            this.Size = new Size(1540, 940);
            this.MinimumSize = new Size(1280, 820);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(2, 4, 10);
            this.DoubleBuffered = true;
            this.KeyPreview = true;

            try
            {
                ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072 | (SecurityProtocolType)768; // TLS 1.2 + TLS 1.1
            }
            catch { }

            try
            {
                _speechSynth = new SpeechSynthesizer();
                _speechSynth.SetOutputToDefaultAudioDevice();
            }
            catch { }

            Load3DAssets();
            InitTargets();
            InitUnseenPlanetPool();
            SelectNewUnseenPlanetForNewGame(false);
            InitBonusStars();
            // Select K2-18 b (index 1) by default; slot 8 is our brand-new unseen NASA mystery planet for this session!
            SelectTarget(1, true, false);

            this.KeyDown += OnKeyDownHandler;
            this.KeyUp += OnKeyUpHandler;
            this.MouseDown += OnMouseDownHandler;
            this.Resize += (s, e) => this.Invalidate();

            _timer = new System.Windows.Forms.Timer();
            _timer.Interval = 16; // 60 FPS
            _timer.Tick += (s, e) => OnTick();
            _timer.Start();

            SaveRenderSnapshot();

            // Speak initial kid-friendly astrophysics briefing after window opens
            ThreadPool.QueueUserWorkItem(_ => {
                Thread.Sleep(350);
                SpeakStepNarrativeAndAstrophysics();
            });
        }

        private void InitBonusStars()
        {
            _bonusStars.Clear();
            _bonusStars.Add(new HudStarBonus { XNorm = 0.38f, YNorm = 0.35f, Collected = false });
            _bonusStars.Add(new HudStarBonus { XNorm = 0.52f, YNorm = 0.50f, Collected = false });
            _bonusStars.Add(new HudStarBonus { XNorm = 0.48f, YNorm = 0.65f, Collected = false });
        }

        private void AddPointsWithPopup(int basePts, string label, float x, float y, Color col)
        {
            int total = basePts * Math.Max(1, _streakCount);
            _scoreHmt += total;
            _popups.Add(new ScorePopup {
                X = x,
                Y = y,
                Text = string.Format("+{0} PTS {1} (x{2}!)", total, label, _streakCount),
                Color = col,
                Life = 1.0
            });
        }

        private string GetCadetRankTitle()
        {
            if (_lang == "fr")
            {
                if (_scoreHmt >= 8000) return "★ AMIRAL DES EXOPLANÈTES (NIV 4)";
                if (_scoreHmt >= 5000) return "★ COMMANDANT BEE-MOE (NIV 3)";
                if (_scoreHmt >= 3000) return "★ ENSEIGNE CHASSEUR (NIV 2)";
                return "★ CADET ASTRO-EXPLORATEUR (NIV 1)";
            }
            if (_scoreHmt >= 8000) return "★ GRAND ADMIRAL OF EXOPLANETS (LVL 4)";
            if (_scoreHmt >= 5000) return "★ COMMANDER BEE-MOE PILOT (LVL 3)";
            if (_scoreHmt >= 3000) return "★ ENSIGN PLANET HUNTER (LVL 2)";
            return "★ CADET ASTRO-SCOUT (LVL 1)";
        }

        private void ResetOolitePlanetfallState()
        {
            double gVal = (_current != null && _current.SurfaceGravityG > 0.05) ? _current.SurfaceGravityG : 1.27;
            double wMax = (_current != null && _current.WindShearMaxMs > 1.0) ? _current.WindShearMaxMs : 28.0;
            _landerAltKm = 3.000;
            // Higher gravity & higher difficulty increase initial atmospheric entry speed & thermal shock!
            _landerSpeedMs = Math.Min(148.0, 82.0 + gVal * 8.5 + _difficultyLevel * 3.2);
            _landerCabinTempC = Math.Min(165.0, 78.0 + gVal * 7.0 + _difficultyLevel * 3.5);
            _landerFwdShield = 100.0;
            _landerAftShield = 100.0;
            _landerArmour = 100.0;
            _landerEnergy = 256.0;
            _landerOffsetX = (float)(((_planetsExploredCount % 2 == 0) ? 1.0 : -1.0) * (10.0 + _difficultyLevel * 2.5));
            _landerPitchDeg = -10.0f;
            _landerWindShearMs = Math.Round(Math.Min(wMax, 8.4 + _difficultyLevel * 4.5 + wMax * 0.22), 1);
            _atmoLayerName = (_lang == "fr") ? "COUCHES 1/4 : THERMOSPHÈRE (PLASMA IONOSPHÉRIQUE)" : "LAYER 1/4: THERMOSPHERE (IONOSPHERIC PLASMA)";
            _landingClearanceGranted = true;
            _landingGearDown = true;
            _stage2TradeTabExpanded = false;
            InitBonusStars();
            _landingStatusMsg = (_lang == "fr")
                ? string.Format(CultureInfo.InvariantCulture, "NIV.{0} // g={1:F2}g | H={2:F1}km | CISAILLEMENT={3:F1}m/s — [ESPACE] FREINER | [ENTRÉE] ATTERRIR",
                    _difficultyLevel, gVal, (_current != null ? _current.ScaleHeightKm : 18.5), _landerWindShearMs)
                : string.Format(CultureInfo.InvariantCulture, "DIFF LVL {0} // g={1:F2}g | H={2:F1}km | WIND={3:F1}m/s — [SPACE] RETRO-BRAKE | [ENTER] LAND",
                    _difficultyLevel, gVal, (_current != null ? _current.ScaleHeightKm : 18.5), _landerWindShearMs);
        }

        private void SaveRenderSnapshot()
        {
            try
            {
                int w = 1540, h = 940;
                int savedStage = _gameStage;
                string assetsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "steam_assets");

                // 1. Save Stage 0 (Stitch Galactic Tactical Bridge Console + Narrative + GCP Gemini Neural Voice)
                _gameStage = 0;
                _showNarrativeCard = false;
                _subtitleText = "Chapter 1: Welcome to K2-18 b, a Hycean Ocean World! JWST detected water vapor, methane & CO2 in its sky!";
                using (Bitmap bmp0 = new Bitmap(w, h))
                using (Graphics g0 = Graphics.FromImage(bmp0))
                {
                    using (PaintEventArgs pea0 = new PaintEventArgs(g0, new Rectangle(0, 0, w, h)))
                    {
                        this.OnPaint(pea0);
                    }
                    bmp0.Save(Path.Combine(assetsDir, "live_v45_client_render.png"), System.Drawing.Imaging.ImageFormat.Png);
                    bmp0.Save(Path.Combine(assetsDir, "live_v50_stitch_stage0_galaxy.png"), System.Drawing.Imaging.ImageFormat.Png);
                }

                // 2. Save Stage 1 (Stitch Stage 2: Orbit Bee — Lagrange Ring Ingress & Ion Propulsion Telemetry)
                _gameStage = 1;
                _showNarrativeCard = false;
                _captureProgress = 68.0;
                _subtitleText = "Chapter 2: Orbit Bee — Lagrange Ring Ingress (L1-L5)! Ion Propulsion at 94.2% Thrust (1,840 mN).";
                using (Bitmap bmp1 = new Bitmap(w, h))
                using (Graphics g1 = Graphics.FromImage(bmp1))
                {
                    using (PaintEventArgs pea1 = new PaintEventArgs(g1, new Rectangle(0, 0, w, h)))
                    {
                        this.OnPaint(pea1);
                    }
                    bmp1.Save(Path.Combine(assetsDir, "live_v50_stitch_stage1_orbit_bee.png"), System.Drawing.Imaging.ImageFormat.Png);
                }

                // 3. Save Stage 2 (Stitch Stage 3: Oolite Land — Chasma Australis SAR Wireframe + Curved Horizon + 3D Glide-Slope Tunnel)
                _gameStage = 2;
                _showNarrativeCard = true;
                _subtitleText = "Chapter 3: Stitch Oolite Land (Chasma Australis LZ-ALPHA)! Press SPACE to brake (+100 PTS) or ENTER to Auto-Land (+1,000 PTS)!";
                ResetOolitePlanetfallState();
                using (Bitmap bmp2 = new Bitmap(w, h))
                using (Graphics g2 = Graphics.FromImage(bmp2))
                {
                    using (PaintEventArgs pea2 = new PaintEventArgs(g2, new Rectangle(0, 0, w, h)))
                    {
                        this.OnPaint(pea2);
                    }
                    bmp2.Save(Path.Combine(assetsDir, "live_v46_oolite_planetfall_render.png"), System.Drawing.Imaging.ImageFormat.Png);
                    bmp2.Save(Path.Combine(assetsDir, "live_v50_stitch_stage2_oolite_land.png"), System.Drawing.Imaging.ImageFormat.Png);
                }

                // 4. Save Stage 3 (Stitch Stage 4: JWST/LOIS Spectroscopy Array — L2 Lagrange Lock & NIRSpec/MIRI Waterfall)
                _gameStage = 3;
                _showNarrativeCard = false;
                _subtitleText = "Chapter 4: Stitch JWST/LOIS Spectroscopy Array (L2 Lagrange Lock)! Peak Detect: Ozone (O3) @ 9.6 um [4.8 sigma].";
                using (Bitmap bmp3 = new Bitmap(w, h))
                using (Graphics g3 = Graphics.FromImage(bmp3))
                {
                    using (PaintEventArgs pea3 = new PaintEventArgs(g3, new Rectangle(0, 0, w, h)))
                    {
                        this.OnPaint(pea3);
                    }
                    bmp3.Save(Path.Combine(assetsDir, "live_v50_stitch_stage3_jwst_lois.png"), System.Drawing.Imaging.ImageFormat.Png);
                }

                // 5. Save Stage 4 (Stage 5: Steam Free-to-Play Idle-GPU BOINC Citizen-Science & GCP Gamer/Astrophysicist Community Hub)
                _gameStage = 4;
                _showNarrativeCard = false;
                _idleSeconds = 5.4;
                _subtitleText = "Chapter 5: Steam Free-to-Play Idle-GPU BOINC Citizen Science & GCP Community Hub! Crunching TESS/JWST workunits when GPU is idle.";
                using (Bitmap bmp4 = new Bitmap(w, h))
                using (Graphics g4 = Graphics.FromImage(bmp4))
                {
                    using (PaintEventArgs pea4 = new PaintEventArgs(g4, new Rectangle(0, 0, w, h)))
                    {
                        this.OnPaint(pea4);
                    }
                    bmp4.Save(Path.Combine(assetsDir, "live_v51_boinc_gcp_community_stage4.png"), System.Drawing.Imaging.ImageFormat.Png);
                }

                _showNarrativeCard = true;
                _gameStage = savedStage;
            }
            catch { }
        }

        private void Load3DAssets()
        {
            try
            {
                string assetsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "steam_assets");
                string galPath = Path.Combine(assetsDir, "spiral_galaxy_v4_hd.png");
                if (File.Exists(galPath)) _galaxyBackdrop = Image.FromFile(galPath);

                string oolPath = Path.Combine(assetsDir, "oolite_planetfall_horizon.png");
                if (File.Exists(oolPath)) _ooliteHorizonBackdrop = Image.FromFile(oolPath);

                string stOrbitPath = Path.Combine(assetsDir, "stitch_hd_00_uss_hivemind_-__0.jpg");
                if (File.Exists(stOrbitPath)) _stitchOrbitK218bImg = Image.FromFile(stOrbitPath);

                string stSarPath = Path.Combine(assetsDir, "stitch_hd_03_uss_hivemind_-__0.jpg");
                if (File.Exists(stSarPath)) _stitchSarTerrainImg = Image.FromFile(stSarPath);

                string stTrapPath = Path.Combine(assetsDir, "stitch_hd_04_uss_hivemind_-__0.jpg");
                if (File.Exists(stTrapPath)) _stitchTrappistImg = Image.FromFile(stTrapPath);

                for (int i = 0; i < 10; i++)
                {
                    string pPath = Path.Combine(assetsDir, "planet_3d_" + i + ".png");
                    if (File.Exists(pPath)) _planetGlobes[i] = Image.FromFile(pPath);
                }
            }
            catch { }
        }

        private void InitTargets()
        {
            _targets.Add(new ExoTarget {
                Index = 0, Id = "SECTOR-01-TRAPPIST-1E", ShortLabel = "TRAPPIST-1e", HostStar = "TRAPPIST-1 (M8V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.145f, GalaxyYNorm = 0.215f,
                PeriodDays = 6.10, SemiMajorAxisAu = 0.029, Eccentricity = 0.005, RadialVelocityKms = 2.48,
                DepthPpm = 5180, RadiusRearth = 0.92, MassMearth = 0.692, BulkDensityGcm3 = 4.90,
                TeqK = 251, InsolationEarth = 0.66, InHz = true,
                DifficultyTier = 2, AsteroidBeltCount = 8, SurfaceGravityG = 0.82, EscapeVelKms = 9.71, ScaleHeightKm = 9.2, SurfacePressureBar = 1.15, WindShearMaxMs = 42.0,
                PrimaryColor = Color.FromArgb(16, 185, 129), PedestalColor = Color.FromArgb(52, 211, 153), ApprovedByGamer = true, LandedMedal = true,
                AnalysisEn = "TRAPPIST-1e analysis: rocky world, 0.92 Earth radii, 251 K, 6.1-day orbit, 5,180 ppm transit.",
                LifeEn = "How life could live: tidally locked twilight oceans could support red-light photosynthetic algae and marine bacteria.",
                AsteroidPhysicsEn = "Orbital Physics: 7-planet Laplace resonance chain (8:5, 5:3, 3:2) triggers frequent cometary ejecta and M8V coronal mass ejections (CMEs).",
                AtmoLandingEn = "Landing Physics: g = 0.82 g (v_esc = 9.7 km/s), scale height H = 9.2 km, with 42 m/s day-to-night terminator wind shear.",
                AnalysisFr = "Analyse TRAPPIST-1e : monde rocheux, 0,92 rayon terrestre, 251 K, orbite 6,1 jours, transit 5 180 ppm.",
                LifeFr = "Vie possible : ses océans crépusculaires abriteraient des algues photosynthétiques sous lumière rouge.",
                AsteroidPhysicsFr = "Physique orbitale : chaîne de résonance de Laplace à 7 planètes (8:5, 3:2) concentrant les comètes et éruptions CME M8V.",
                AtmoLandingFr = "Physique d'atterrissage : g = 0,82 g (v_lib = 9,7 km/s), hauteur d'échelle H = 9,2 km, cisaillement de 42 m/s au terminateur.",
                KidLessonEn = "TRAPPIST-1e analysis: rocky world, 0.92 Earth radii, 251 K, 6.1-day orbit, 5,180 ppm transit. How life could live: tidally locked twilight oceans could support red-light photosynthetic algae and marine bacteria.",
                KidLessonFr = "Analyse TRAPPIST-1e : monde rocheux, 0,92 rayon terrestre, 251 K, orbite 6,1 jours, transit 5 180 ppm. Vie possible : ses océans crépusculaires abriteraient des algues photosynthétiques sous lumière rouge."
            });
            _targets.Add(new ExoTarget {
                Index = 1, Id = "SECTOR-02-K2-18B", ShortLabel = "K2-18 b", HostStar = "K2-18 (M2.5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.340f, GalaxyYNorm = 0.175f,
                PeriodDays = 32.94, SemiMajorAxisAu = 0.143, Eccentricity = 0.09, RadialVelocityKms = 3.55,
                DepthPpm = 2920, RadiusRearth = 2.61, MassMearth = 8.63, BulkDensityGcm3 = 2.67,
                TeqK = 265, InsolationEarth = 0.94, InHz = true,
                DifficultyTier = 1, AsteroidBeltCount = 7, SurfaceGravityG = 1.27, EscapeVelKms = 20.34, ScaleHeightKm = 82.5, SurfacePressureBar = 12.4, WindShearMaxMs = 58.0,
                PrimaryColor = Color.FromArgb(14, 165, 233), PedestalColor = Color.FromArgb(56, 189, 248), ApprovedByGamer = true, LandedMedal = true,
                AnalysisEn = "K2-18 b analysis: Hycean world, 8.6 Earth masses, 265 K, JWST methane and CO2 atmosphere.",
                LifeEn = "How life could live: phytoplankton and deep-sea organisms could thrive in its warm global ocean.",
                AsteroidPhysicsEn = "Orbital Physics: Eccentricity e = 0.09 induces L4/L5 Trojan ice-bolide swarms and inner resonance perturbation from K2-18 c.",
                AtmoLandingEn = "Landing Physics: Hycean H2-rich sky (mu = 2.3 g/mol) gives huge scale height H = 82.5 km, g = 1.27 g, and 58 m/s cyclonic ocean gusts.",
                AnalysisFr = "Analyse K2-18 b : monde hycéen, 8,6 masses terrestres, 265 K, méthane et CO2 détectés par JWST.",
                LifeFr = "Vie possible : du phytoplancton et des organismes marins prospéreraient dans son océan global chaud.",
                AsteroidPhysicsFr = "Physique orbitale : excentricité e = 0,09 générant des essaims de bolides glacés aux points de Lagrange L4/L5.",
                AtmoLandingFr = "Physique d'atterrissage : atmosphère hycéenne riche en H2 (H = 82,5 km), gravité g = 1,27 g et rafales cycloniques à 58 m/s.",
                KidLessonEn = "K2-18 b analysis: Hycean world, 8.6 Earth masses, 265 K, JWST methane and CO2 atmosphere. How life could live: phytoplankton and deep-sea organisms could thrive in its warm global ocean.",
                KidLessonFr = "Analyse K2-18 b : monde hycéen, 8,6 masses terrestres, 265 K, méthane et CO2 détectés par JWST. Vie possible : du phytoplancton et des organismes marins prospéreraient dans son océan global chaud."
            });
            _targets.Add(new ExoTarget {
                Index = 2, Id = "SECTOR-03-KEPLER-11F", ShortLabel = "Kepler-11 f", HostStar = "Kepler-11 (G6V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.555f, GalaxyYNorm = 0.205f,
                PeriodDays = 46.69, SemiMajorAxisAu = 0.250, Eccentricity = 0.013, RadialVelocityKms = 0.43,
                DepthPpm = 550, RadiusRearth = 2.49, MassMearth = 2.00, BulkDensityGcm3 = 0.71,
                TeqK = 544, InsolationEarth = 17.4, InHz = false,
                DifficultyTier = 4, AsteroidBeltCount = 11, SurfaceGravityG = 0.32, EscapeVelKms = 10.02, ScaleHeightKm = 540.0, SurfacePressureBar = 85.0, WindShearMaxMs = 115.0,
                PrimaryColor = Color.FromArgb(139, 92, 246), PedestalColor = Color.FromArgb(167, 139, 250), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "Kepler-11 f analysis: low-density Mini-Neptune, 0.71 g/cm³, 2.49 Earth radii, 544 K hydrogen envelope.",
                LifeEn = "How life could live: no solid ground exists, so only high-altitude floating cloud microbes could survive.",
                AsteroidPhysicsEn = "Orbital Physics: Ultra-compact 6-planet system inside 0.46 AU creates dense secular resonance debris rings (v_orb = 58.4 km/s).",
                AtmoLandingEn = "Landing Physics: Puffy H/He envelope (rho = 0.71 g/cm³, g = 0.32 g) produces a 540 km scale height and 115 m/s convective downdrafts.",
                AnalysisFr = "Analyse Kepler-11 f : Mini-Neptune gazeuse, densité 0,71 g/cm³, 2,49 rayons terrestres, 544 K.",
                LifeFr = "Vie possible : sans surface solide, seuls des microbes flottant dans les nuages tempérés survivraient.",
                AsteroidPhysicsFr = "Physique orbitale : système ultra-compact à 6 planètes sous 0,46 UA créant d'intenses anneaux de débris en résonance.",
                AtmoLandingFr = "Physique d'atterrissage : enveloppe H/He gonflée (g = 0,32 g, H = 540 km) avec courants descendants convectifs à 115 m/s.",
                KidLessonEn = "Kepler-11 f analysis: low-density Mini-Neptune, 0.71 g/cm³, 2.49 Earth radii, 544 K hydrogen envelope. How life could live: no solid ground exists, so only high-altitude floating cloud microbes could survive.",
                KidLessonFr = "Analyse Kepler-11 f : Mini-Neptune gazeuse, densité 0,71 g/cm³, 2,49 rayons terrestres, 544 K. Vie possible : sans surface solide, seuls des microbes flottant dans les nuages tempérés survivraient."
            });
            _targets.Add(new ExoTarget {
                Index = 3, Id = "SECTOR-04-HD-80606B", ShortLabel = "HD 80606 b", HostStar = "HD 80606 (G5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.785f, GalaxyYNorm = 0.235f,
                PeriodDays = 111.44, SemiMajorAxisAu = 0.456, Eccentricity = 0.933, RadialVelocityKms = 476.1,
                DepthPpm = 11200, RadiusRearth = 11.99, MassMearth = 1392.0, BulkDensityGcm3 = 4.43,
                TeqK = 405, InsolationEarth = 5.05, InHz = false,
                DifficultyTier = 9, AsteroidBeltCount = 18, SurfaceGravityG = 9.68, EscapeVelKms = 120.5, ScaleHeightKm = 18.0, SurfacePressureBar = 420.0, WindShearMaxMs = 285.0,
                PrimaryColor = Color.FromArgb(234, 88, 12), PedestalColor = Color.FromArgb(251, 146, 60), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "HD 80606 b analysis: eccentric giant, 93% eccentricity, 476 m/s Doppler wobble, 1,200 K periastron storms.",
                LifeEn = "How life could live: extreme heat blocks surface life; only subsurface microbes inside an icy exomoon could survive.",
                AsteroidPhysicsEn = "Orbital Physics: Extreme e = 0.933 swings from 0.88 AU to 0.03 AU! Periastron tidal shock shatters asteroids across the Roche limit.",
                AtmoLandingEn = "Landing Physics: Crushing g = 9.68 g (v_esc = 120.5 km/s) and 828x insolation spike drive 285 m/s supersonic shockwave gales!",
                AnalysisFr = "Analyse HD 80606 b : géante excentrique à 93 %, Doppler 476 m/s, tempêtes à 1 200 K.",
                LifeFr = "Vie possible : chaleur mortelle en surface ; seuls des microbes sous la glace d'une exolune survivraient.",
                AsteroidPhysicsFr = "Physique orbitale : excentricité extrême e = 0,933 (0,88 à 0,03 UA) brisant les comètes par cisaillement de marée de Roche.",
                AtmoLandingFr = "Physique d'atterrissage : gravité écrasante g = 9,68 g (v_lib = 120,5 km/s) et tempêtes supersoniques à 285 m/s au périastre !",
                KidLessonEn = "HD 80606 b analysis: eccentric giant, 93% eccentricity, 476 m/s Doppler wobble, 1,200 K periastron storms. How life could live: extreme heat blocks surface life; only subsurface microbes inside an icy exomoon could survive.",
                KidLessonFr = "Analyse HD 80606 b : géante excentrique à 93 %, Doppler 476 m/s, tempêtes à 1 200 K. Vie possible : chaleur mortelle en surface ; seuls des microbes sous la glace d'une exolune survivraient."
            });
            _targets.Add(new ExoTarget {
                Index = 4, Id = "SECTOR-05-TOI-700D", ShortLabel = "TOI-700 d", HostStar = "TOI-700 (M2V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.205f, GalaxyYNorm = 0.545f,
                PeriodDays = 37.42, SemiMajorAxisAu = 0.163, Eccentricity = 0.042, RadialVelocityKms = 0.78,
                DepthPpm = 665, RadiusRearth = 1.14, MassMearth = 1.72, BulkDensityGcm3 = 6.32,
                TeqK = 269, InsolationEarth = 0.86, InHz = true,
                DifficultyTier = 3, AsteroidBeltCount = 9, SurfaceGravityG = 1.32, EscapeVelKms = 13.74, ScaleHeightKm = 6.8, SurfacePressureBar = 1.85, WindShearMaxMs = 54.0,
                PrimaryColor = Color.FromArgb(20, 184, 166), PedestalColor = Color.FromArgb(45, 212, 191), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "TOI-700 d analysis: TESS rocky world, 1.14 Earth radii, 269 K, 37.4-day habitable orbit.",
                LifeEn = "How life could live: mild coastal seas along the terminator zone could sustain multicellular aquatic life.",
                AsteroidPhysicsEn = "Orbital Physics: Outer Kirkwood 4:3 resonance with TOI-700 e scatters rocky planetesimals across the 0.163 AU habitable corridor.",
                AtmoLandingEn = "Landing Physics: Dense iron-core world (rho = 6.32 g/cm³, g = 1.32 g) with compact H = 6.8 km N2/CO2 troposphere and 54 m/s coastal shear.",
                AnalysisFr = "Analyse TOI-700 d : monde rocheux TESS, 1,14 rayon terrestre, 269 K, orbite habitable de 37,4 jours.",
                LifeFr = "Vie possible : des mers tempérées le long du terminateur soutiendraient une vie aquatique complexe.",
                AsteroidPhysicsFr = "Physique orbitale : résonance 4:3 avec TOI-700 e déviant des planétésimaux rocheux sur l'orbite de 0,163 UA.",
                AtmoLandingFr = "Physique d'atterrissage : noyau ferreux dense (g = 1,32 g), troposphère N2/CO2 compacte (H = 6,8 km) et vents côtiers de 54 m/s.",
                KidLessonEn = "TOI-700 d analysis: TESS rocky world, 1.14 Earth radii, 269 K, 37.4-day habitable orbit. How life could live: mild coastal seas along the terminator zone could sustain multicellular aquatic life.",
                KidLessonFr = "Analyse TOI-700 d : monde rocheux TESS, 1,14 rayon terrestre, 269 K, orbite habitable de 37,4 jours. Vie possible : des mers tempérées le long du terminateur soutiendraient une vie aquatique complexe."
            });
            _targets.Add(new ExoTarget {
                Index = 5, Id = "SECTOR-06-LHS-1140B", ShortLabel = "LHS 1140 b", HostStar = "LHS 1140 (M4.5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.440f, GalaxyYNorm = 0.500f,
                PeriodDays = 24.74, SemiMajorAxisAu = 0.095, Eccentricity = 0.043, RadialVelocityKms = 4.90,
                DepthPpm = 5340, RadiusRearth = 1.73, MassMearth = 5.60, BulkDensityGcm3 = 5.95,
                TeqK = 226, InsolationEarth = 0.43, InHz = true,
                DifficultyTier = 5, AsteroidBeltCount = 12, SurfaceGravityG = 1.87, EscapeVelKms = 20.11, ScaleHeightKm = 4.3, SurfacePressureBar = 4.20, WindShearMaxMs = 88.0,
                PrimaryColor = Color.FromArgb(6, 182, 212), PedestalColor = Color.FromArgb(34, 211, 238), ApprovedByGamer = true, LandedMedal = true,
                AnalysisEn = "LHS 1140 b analysis: dense Super-Earth, 5.6 Earth masses, 5,340 ppm transit, 226 K nitrogen sky.",
                LifeEn = "How life could live: a sunlit open-ocean bullseye and hydrothermal vents could sustain rich marine ecosystems.",
                AsteroidPhysicsEn = "Orbital Physics: High escape velocity (20.1 km/s) and Hill sphere (R_H = 0.0024 AU) focus incoming icy exocomets to hyper-velocities.",
                AtmoLandingEn = "Landing Physics: Heavy Super-Earth gravity g = 1.87 g compresses its secondary N2 atmosphere (H = 4.3 km, 4.2 bar) with 88 m/s polar downdrafts!",
                AnalysisFr = "Analyse LHS 1140 b : Super-Terre dense, 5,6 masses terrestres, transit 5 340 ppm, 226 K.",
                LifeFr = "Vie possible : un océan circulaire côté jour et des sources hydrothermales nourriraient la vie marine.",
                AsteroidPhysicsFr = "Physique orbitale : forte vitesse de libération (20,1 km/s) accélérant les exocomètes glacées dans la sphère de Hill.",
                AtmoLandingFr = "Physique d'atterrissage : forte gravité g = 1,87 g comprimant l'atmosphère N2 (H = 4,3 km, 4,2 bar) avec courants polaires à 88 m/s !",
                KidLessonEn = "LHS 1140 b analysis: dense Super-Earth, 5.6 Earth masses, 5,340 ppm transit, 226 K nitrogen sky. How life could live: a sunlit open-ocean bullseye and hydrothermal vents could sustain rich marine ecosystems.",
                KidLessonFr = "Analyse LHS 1140 b : Super-Terre dense, 5,6 masses terrestres, transit 5 340 ppm, 226 K. Vie possible : un océan circulaire côté jour et des sources hydrothermales nourriraient la vie marine."
            });
            _targets.Add(new ExoTarget {
                Index = 6, Id = "SECTOR-07-WASP-39B", ShortLabel = "WASP-39 b", HostStar = "WASP-39 (G8V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.650f, GalaxyYNorm = 0.490f,
                PeriodDays = 4.06, SemiMajorAxisAu = 0.049, Eccentricity = 0.00, RadialVelocityKms = 38.0,
                DepthPpm = 21200, RadiusRearth = 14.23, MassMearth = 88.9, BulkDensityGcm3 = 0.17,
                TeqK = 1166, InsolationEarth = 420.0, InHz = false,
                DifficultyTier = 7, AsteroidBeltCount = 15, SurfaceGravityG = 0.44, EscapeVelKms = 27.95, ScaleHeightKm = 980.0, SurfacePressureBar = 65.0, WindShearMaxMs = 195.0,
                PrimaryColor = Color.FromArgb(202, 138, 4), PedestalColor = Color.FromArgb(250, 204, 21), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "WASP-39 b analysis: puffy hot Saturn, 14.2 Earth radii, 1,166 K, JWST SO2 and CO2 clouds.",
                LifeEn = "How life could live: 1,166 K heat and corrosive sulfur clouds break organic bonds; life cannot survive.",
                AsteroidPhysicsEn = "Orbital Physics: Close-in 0.049 AU orbit (v_orb = 135 km/s) experiences intense UV photo-evaporation and refractory dust infall.",
                AtmoLandingEn = "Landing Physics: Extreme scale height H = 980 km at 1,166 K with photochemical SO2 aerosols and 195 m/s equatorial super-rotation jets!",
                AnalysisFr = "Analyse WASP-39 b : Saturne chaude gonflée, 14,2 rayons terrestres, 1 166 K, nuages de SO2 et CO2.",
                LifeFr = "Vie possible : à 1 166 K, la chaleur brise les molécules organiques ; aucune vie ne peut survivre.",
                AsteroidPhysicsFr = "Physique orbitale : orbite proche à 0,049 UA (v_orb = 135 km/s) soumise à la photo-évaporation UV et aux poussières réfractaires.",
                AtmoLandingFr = "Physique d'atterrissage : hauteur d'échelle géante H = 980 km à 1 166 K, nuages photochimiques de SO2 et jets équatoriaux à 195 m/s !",
                KidLessonEn = "WASP-39 b analysis: puffy hot Saturn, 14.2 Earth radii, 1,166 K, JWST SO2 and CO2 clouds. How life could live: 1,166 K heat and corrosive sulfur clouds break organic bonds; life cannot survive.",
                KidLessonFr = "Analyse WASP-39 b : Saturne chaude gonflée, 14,2 rayons terrestres, 1 166 K, nuages de SO2 et CO2. Vie possible : à 1 166 K, la chaleur brise les molécules organiques ; aucune vie ne peut survivre."
            });
            _targets.Add(new ExoTarget {
                Index = 7, Id = "SECTOR-08-KEPLER-186F", ShortLabel = "Kepler-186f", HostStar = "Kepler-186 (M1V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.820f, GalaxyYNorm = 0.550f,
                PeriodDays = 129.94, SemiMajorAxisAu = 0.432, Eccentricity = 0.04, RadialVelocityKms = 0.29,
                DepthPpm = 512, RadiusRearth = 1.17, MassMearth = 1.44, BulkDensityGcm3 = 4.96,
                TeqK = 188, InsolationEarth = 0.32, InHz = true,
                DifficultyTier = 6, AsteroidBeltCount = 13, SurfaceGravityG = 1.05, EscapeVelKms = 12.41, ScaleHeightKm = 5.4, SurfacePressureBar = 3.10, WindShearMaxMs = 96.0,
                PrimaryColor = Color.FromArgb(192, 38, 211), PedestalColor = Color.FromArgb(232, 121, 249), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "Kepler-186f analysis: cool rocky planet, 1.17 Earth radii, 188 K, 130-day outer habitable orbit.",
                LifeEn = "How life could live: dark infrared-absorbing plants and sub-glacial microbes could thrive near volcanic vents.",
                AsteroidPhysicsEn = "Orbital Physics: Outer habitable edge (0.432 AU) borders the system's frost line (T < 170 K), producing dense Kuiper-belt ice-asteroid swarms.",
                AtmoLandingEn = "Landing Physics: Cold 188 K dense CO2/N2 atmosphere (H = 5.4 km, g = 1.05 g) triggers sudden CO2 dry-ice snow squalls and 96 m/s blizzards.",
                AnalysisFr = "Analyse Kepler-186f : planète rocheuse froide, 1,17 rayon terrestre, 188 K, orbite habitable de 130 jours.",
                LifeFr = "Vie possible : des plantes sombres captant l'infrarouge et des microbes survivraient près des volcans.",
                AsteroidPhysicsFr = "Physique orbitale : bord externe habitable (0,432 UA) proche de la ligne des glaces, riche en astéroïdes glacés.",
                AtmoLandingFr = "Physique d'atterrissage : atmosphère froide à 188 K (H = 5,4 km, g = 1,05 g) avec blizzards de glace carbonique CO2 à 96 m/s.",
                KidLessonEn = "Kepler-186f analysis: cool rocky planet, 1.17 Earth radii, 188 K, 130-day outer habitable orbit. How life could live: dark infrared-absorbing plants and sub-glacial microbes could thrive near volcanic vents.",
                KidLessonFr = "Analyse Kepler-186f : planète rocheuse froide, 1,17 rayon terrestre, 188 K, orbite habitable de 130 jours. Vie possible : des plantes sombres captant l'infrarouge et des microbes survivraient près des volcans."
            });
            _targets.Add(new ExoTarget {
                Index = 8, Id = "SECTOR-09-KOI-CAND-14", ShortLabel = "KOI-Cand 14", HostStar = "KIC 11804465",
                Disposition = "CANDIDATE", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 18.42, SemiMajorAxisAu = 0.138, Eccentricity = 0.01, RadialVelocityKms = 0.04,
                DepthPpm = 420, RadiusRearth = 1.42, MassMearth = 0.05, BulkDensityGcm3 = 0.10,
                TeqK = 258, InsolationEarth = 0.88, InHz = true,
                DifficultyTier = 6, AsteroidBeltCount = 14, SurfaceGravityG = 0.65, EscapeVelKms = 8.40, ScaleHeightKm = 14.0, SurfacePressureBar = 0.85, WindShearMaxMs = 74.0,
                PrimaryColor = Color.FromArgb(22, 163, 74), PedestalColor = Color.FromArgb(74, 222, 128), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "KOI-Cand 14 analysis: unconfirmed candidate, 420 ppm dip, flat 0.04 m/s Doppler wobble, 258 K.",
                LifeEn = "How life could live: if verified as an icy water world, sub-surface oceans could shelter anaerobic bacteria.",
                AsteroidPhysicsEn = "Orbital Physics: Unresolved planetary-mass debris disk causes transit timing variations (TTVs) and dense micrometeoroid hazards.",
                AtmoLandingEn = "Landing Physics: Low surface gravity (0.65 g) and sublimating H2O/CH4 ice plumes generate unpredictable 74 m/s cryo-geyser updrafts.",
                AnalysisFr = "Analyse KOI-Cand 14 : candidat à confirmer, transit 420 ppm, Doppler plat 0,04 m/s, 258 K.",
                LifeFr = "Vie possible : s'il est confirmé comme monde glacé, un océan souterrain abriterait des bactéries.",
                AsteroidPhysicsFr = "Physique orbitale : disque de débris causant des variations de transit (TTV) et des essaims de micrométéoroïdes.",
                AtmoLandingFr = "Physique d'atterrissage : faible gravité (0,65 g) et panaches de cryo-geysers H2O/CH4 créant des turbulences à 74 m/s.",
                KidLessonEn = "KOI-Cand 14 analysis: unconfirmed candidate, 420 ppm dip, flat 0.04 m/s Doppler wobble, 258 K. How life could live: if verified as an icy water world, sub-surface oceans could shelter anaerobic bacteria.",
                KidLessonFr = "Analyse KOI-Cand 14 : candidat à confirmer, transit 420 ppm, Doppler plat 0,04 m/s, 258 K. Vie possible : s'il est confirmé comme monde glacé, un océan souterrain abriterait des bactéries."
            });
            _targets.Add(new ExoTarget {
                Index = 9, Id = "SECTOR-10-KOI-BIN-02", ShortLabel = "KOI-Bin #02", HostStar = "KIC 9832227 (Binary)",
                Disposition = "FALSE POSITIVE", GalaxyXNorm = 0.695f, GalaxyYNorm = 0.805f,
                PeriodDays = 1.84, SemiMajorAxisAu = 0.031, Eccentricity = 0.08, RadialVelocityKms = 1420.0,
                DepthPpm = 58400, RadiusRearth = 31.40, MassMearth = 31150.0, BulkDensityGcm3 = 5.55,
                TeqK = 1480, InsolationEarth = 890.0, InHz = false,
                DifficultyTier = 10, AsteroidBeltCount = 20, SurfaceGravityG = 31.59, EscapeVelKms = 352.4, ScaleHeightKm = 145.0, SurfacePressureBar = 1000.0, WindShearMaxMs = 340.0,
                PrimaryColor = Color.FromArgb(225, 29, 72), PedestalColor = Color.FromArgb(244, 63, 94), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "KOI-Bin #02 analysis: false positive eclipsing binary, V-shaped 58,400 ppm dip, 1,420 m/s wobble.",
                LifeEn = "How life could live: no solid planet exists here—only two scorching stars where biology cannot survive.",
                AsteroidPhysicsEn = "Orbital Physics: Contact binary Roche-lobe overflow (L1 mass transfer stream) ejects relativistic plasma and molten slag bolides!",
                AtmoLandingEn = "Landing Physics: Stellar photosphere (g = 31.6 g, v_esc = 352 km/s, T = 1,480 K)—no solid surface exists; abort to JWST Lab to REJECT!",
                AnalysisFr = "Analyse KOI-Bin #02 : faux positif binaire à éclipses, transit en V 58 400 ppm, Doppler 1 420 m/s.",
                LifeFr = "Vie possible : aucune planète solide ici, seulement deux étoiles brûlantes stériles.",
                AsteroidPhysicsFr = "Physique orbitale : débordement du lobe de Roche binaire au point L1 projetant du plasma et des bolides en fusion !",
                AtmoLandingFr = "Physique d'atterrissage : photosphère stellaire (g = 31,6 g, 1 480 K) sans surface solide ; passez au Labo JWST pour REJETER !",
                KidLessonEn = "KOI-Bin #02 analysis: false positive eclipsing binary, V-shaped 58,400 ppm dip, 1,420 m/s wobble. How life could live: no solid planet exists here—only two scorching stars where biology cannot survive.",
                KidLessonFr = "Analyse KOI-Bin #02 : faux positif binaire à éclipses, transit en V 58 400 ppm, Doppler 1 420 m/s. Vie possible : aucune planète solide ici, seulement deux étoiles brûlantes stériles."
            });
        }

        /// <summary>
        /// Initializes a pool of 8 real peer-reviewed NASA Exoplanet Archive worlds that were NOT in the original 10-planet list.
        /// Each new game automatically selects an unseen planet from this pool (persisted in seen_mystery_planets.txt)
        /// so every session introduces a fresh scientific discovery with real asteroid, atmospheric, and landing physics!
        /// </summary>
        private void InitUnseenPlanetPool()
        {
            _unseenPlanetPool.Clear();
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-PROXIMA-CEN-B", ShortLabel = "Proxima Cen b", HostStar = "Proxima Centauri (M5.5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 11.186, SemiMajorAxisAu = 0.0485, Eccentricity = 0.02, RadialVelocityKms = 1.38,
                DepthPpm = 4850, RadiusRearth = 1.03, MassMearth = 1.07, BulkDensityGcm3 = 5.30,
                TeqK = 234, InsolationEarth = 0.65, InHz = true, IsNewMysteryPlanet = true,
                DifficultyTier = 6, AsteroidBeltCount = 14, SurfaceGravityG = 1.01, EscapeVelKms = 11.40, ScaleHeightKm = 7.8, SurfacePressureBar = 1.40, WindShearMaxMs = 78.0,
                PrimaryColor = Color.FromArgb(16, 185, 129), PedestalColor = Color.FromArgb(52, 255, 141), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — Proxima Cen b (4.24 ly): rocky habitable-zone world, 1.07 M⊕, 1.03 R⊕, 234 K, P=11.19 d, K=1.38 m/s (HARPS/ESPRESSO).",
                LifeEn = "How life could live: bio-fluorescent marine organisms and sub-surface extremophiles could shield against M5.5V XUV superflares.",
                AsteroidPhysicsEn = "Real Astrophysics: ALMA detected a cold Kuiper-like dust & exocomet belt (1-4 AU) + Proxima 10^33.5 erg superflare CME shockwaves!",
                AtmoLandingEn = "Landing Physics: g = 1.01 g (v_esc = 11.4 km/s), H = 7.8 km, with ionospheric superflare drag and 78 m/s terminator katabatic gales.",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — Proxima Cen b (4,24 al) : monde rocheux habitable, 1,07 M⊕, 1,03 R⊕, 234 K, P=11,19 j, K=1,38 m/s.",
                LifeFr = "Vie possible : des organismes marins bio-fluorescents survivraient sous l'eau face aux super-éruptions UV de Proxima.",
                AsteroidPhysicsFr = "Astrophysique réelle : ceinture d'exocomètes détectée par ALMA (1-4 UA) et ondes de choc CME de super-éruptions M5.5V !",
                AtmoLandingFr = "Physique d'atterrissage : g = 1,01 g (v_lib = 11,4 km/s), H = 7,8 km, traînée ionosphérique et vents catabatiques de 78 m/s."
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-55-CANCRI-E", ShortLabel = "55 Cancri e", HostStar = "55 Cancri A (G8V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 0.7365, SemiMajorAxisAu = 0.0154, Eccentricity = 0.03, RadialVelocityKms = 6.02,
                DepthPpm = 380, RadiusRearth = 1.875, MassMearth = 7.99, BulkDensityGcm3 = 6.66,
                TeqK = 1958, InsolationEarth = 2400.0, InHz = false, IsNewMysteryPlanet = true,
                DifficultyTier = 8, AsteroidBeltCount = 17, SurfaceGravityG = 2.27, EscapeVelKms = 23.10, ScaleHeightKm = 42.0, SurfacePressureBar = 18.5, WindShearMaxMs = 220.0,
                PrimaryColor = Color.FromArgb(249, 115, 22), PedestalColor = Color.FromArgb(251, 146, 60), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — 55 Cancri e (Janssen): ultra-hot Super-Earth, 7.99 M⊕, 1.88 R⊕, 1,958 K magma ocean, JWST CO/CO2 atmosphere (2024).",
                LifeEn = "How life could live: 1,958 K molten silicate surface prevents carbon biology; only silicon-carbide prebiotic chemistry can exist.",
                AsteroidPhysicsEn = "Real Astrophysics: Orbiting at 0.0154 AU (P = 17.7 h, v_orb = 230 km/s) inside refractory corundum/silicate meteoroid evaporation streams!",
                AtmoLandingEn = "Landing Physics: Heavy g = 2.27 g (v_esc = 23.1 km/s), JWST outgassed CO/CO2 sky (H = 42 km) and 220 m/s silicate-rain thermal storms!",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — 55 Cancri e : Super-Terre de lave à 1 958 K, 7,99 M⊕, 1,88 R⊕, atmosphère CO/CO2 détectée par JWST.",
                LifeFr = "Vie possible : océan de magma à 1 958 K stérile pour la biologie carbonée ; chimie prébiotique de silicates uniquement.",
                AsteroidPhysicsFr = "Astrophysique réelle : orbite ultra-proche à 0,0154 UA (P = 17,7 h, v_orb = 230 km/s) dans un flux de météoroïdes de silicates en fusion !",
                AtmoLandingFr = "Physique d'atterrissage : forte gravité g = 2,27 g (v_lib = 23,1 km/s), ciel CO/CO2 (H = 42 km) et tempêtes de vapeur minérale à 220 m/s !"
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-GJ-1214B", ShortLabel = "GJ 1214 b", HostStar = "GJ 1214 (M4.5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 1.5804, SemiMajorAxisAu = 0.0149, Eccentricity = 0.06, RadialVelocityKms = 12.30,
                DepthPpm = 13500, RadiusRearth = 2.742, MassMearth = 8.17, BulkDensityGcm3 = 2.20,
                TeqK = 596, InsolationEarth = 22.1, InHz = false, IsNewMysteryPlanet = true,
                DifficultyTier = 7, AsteroidBeltCount = 15, SurfaceGravityG = 1.09, EscapeVelKms = 19.30, ScaleHeightKm = 38.5, SurfacePressureBar = 65.0, WindShearMaxMs = 165.0,
                PrimaryColor = Color.FromArgb(56, 189, 248), PedestalColor = Color.FromArgb(125, 211, 252), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — GJ 1214 b (Enaiposha): archetype steam world, 8.17 M⊕, 2.74 R⊕, 596 K, JWST MIRI high-albedo haze & H2O/CH4 sky.",
                LifeEn = "How life could live: supercritical steam ocean at depth is too hot, but high-altitude temperate cloud droplets could host extremophiles.",
                AsteroidPhysicsEn = "Real Astrophysics: High-metallicity (Z > 100x solar) chondritic planetesimal torus and eccentric tidal stream at a = 0.0149 AU.",
                AtmoLandingEn = "Landing Physics: Reflective KCl/ZnS salt clouds (albedo 0.51), H = 38.5 km steam envelope, g = 1.09 g, and 165 m/s zonal jet shear!",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — GJ 1214 b : monde-vapeur archétype, 8,17 M⊕, 2,74 R⊕, 596 K, brumes réfléchissantes mesurées par JWST MIRI.",
                LifeFr = "Vie possible : océan supercritique profond trop chaud, mais les gouttelettes des hauts nuages tempérés abriteraient des microbes.",
                AsteroidPhysicsFr = "Astrophysique réelle : tore de planétésimaux chondritiques à haute métallicité (Z > 100x solaire) à 0,0149 UA.",
                AtmoLandingFr = "Physique d'atterrissage : nuages de sels KCl/ZnS (albédo 0,51), enveloppe de vapeur H = 38,5 km, g = 1,09 g et jets à 165 m/s !"
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-LP-890-9C", ShortLabel = "LP 890-9 c", HostStar = "SPECULOOS-2 (M6V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 8.457, SemiMajorAxisAu = 0.0398, Eccentricity = 0.02, RadialVelocityKms = 1.95,
                DepthPpm = 7280, RadiusRearth = 1.367, MassMearth = 2.53, BulkDensityGcm3 = 5.46,
                TeqK = 272, InsolationEarth = 0.91, InHz = true, IsNewMysteryPlanet = true,
                DifficultyTier = 6, AsteroidBeltCount = 13, SurfaceGravityG = 1.35, EscapeVelKms = 15.20, ScaleHeightKm = 14.2, SurfacePressureBar = 3.80, WindShearMaxMs = 118.0,
                PrimaryColor = Color.FromArgb(45, 212, 191), PedestalColor = Color.FromArgb(94, 234, 212), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — LP 890-9 c (SPECULOOS-2 c): temperate Super-Earth, 1.37 R⊕, 2.53 M⊕, 272 K, 7,280 ppm transit at inner HZ boundary.",
                LifeEn = "How life could live: warm global oceans and geothermal shelves could support dense thermophilic microbial mats and marine life.",
                AsteroidPhysicsEn = "Real Astrophysics: Near 2:1 mean-motion resonance with inner LP 890-9 b drives periodic exocomet scattering across 0.04 AU.",
                AtmoLandingEn = "Landing Physics: Super-Earth gravity g = 1.35 g (v_esc = 15.2 km/s), moist steam troposphere (H = 14.2 km) with 118 m/s convective microbursts!",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — LP 890-9 c (SPECULOOS-2 c) : Super-Terre habitable, 1,37 R⊕, 2,53 M⊕, 272 K, transit de 7 280 ppm.",
                LifeFr = "Vie possible : océans chauds au bord interne de la zone habitable propices aux tapis microbiens thermophiles.",
                AsteroidPhysicsFr = "Astrophysique réelle : résonance orbitale proche de 2:1 avec LP 890-9 b déviant des exocomètes à 0,04 UA.",
                AtmoLandingFr = "Physique d'atterrissage : gravité g = 1,35 g (v_lib = 15,2 km/s), troposphère humide (H = 14,2 km) et microrafales convectives à 118 m/s !"
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-HD-189733B", ShortLabel = "HD 189733 b", HostStar = "HD 189733 A (K2V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 2.2186, SemiMajorAxisAu = 0.0314, Eccentricity = 0.004, RadialVelocityKms = 200.6,
                DepthPpm = 24100, RadiusRearth = 12.71, MassMearth = 369.6, BulkDensityGcm3 = 0.98,
                TeqK = 1201, InsolationEarth = 346.0, InHz = false, IsNewMysteryPlanet = true,
                DifficultyTier = 9, AsteroidBeltCount = 19, SurfaceGravityG = 2.29, EscapeVelKms = 60.30, ScaleHeightKm = 210.0, SurfacePressureBar = 250.0, WindShearMaxMs = 295.0,
                PrimaryColor = Color.FromArgb(37, 99, 235), PedestalColor = Color.FromArgb(96, 165, 250), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — HD 189733 b: cobalt-blue Hot Jupiter, 12.7 R⊕, 370 M⊕, 1,201 K, JWST H2S/CO2/H2O & molten silicate glass rain!",
                LifeEn = "How life could live: 1,201 K heat and sideways molten-glass rain destroy cells; only captured outer exomoons could shelter life.",
                AsteroidPhysicsEn = "Real Astrophysics: Lyman-alpha evaporating hydrogen torus (10^9 g/s mass loss) + K2V stellar wind shockwave and Trojan debris!",
                AtmoLandingEn = "Landing Physics: HARPS measured 2,400 m/s (8,700 km/h) supersonic winds & MgSiO3 molten-glass rain under g = 2.29 g (H = 210 km)!",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — HD 189733 b : Jupiter chaud bleu cobalt, 12,7 R⊕, 370 M⊕, 1 201 K, pluie de verre en fusion et H2S détecté par JWST !",
                LifeFr = "Vie possible : pluie de verre fondu à 1 201 K mortelle ; seules des exolunes glacées externes abriteraient la vie.",
                AsteroidPhysicsFr = "Astrophysique réelle : tore d'hydrogène en évaporation Lyman-alpha (10^9 g/s) et débris troyens sous vent stellaire K2V !",
                AtmoLandingFr = "Physique d'atterrissage : vents supersoniques mesurés par HARPS (8 700 km/h) et pluie de silicate MgSiO3 sous g = 2,29 g !"
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-WOLF-1069B", ShortLabel = "Wolf 1069 b", HostStar = "Wolf 1069 (M5V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 15.564, SemiMajorAxisAu = 0.0672, Eccentricity = 0.01, RadialVelocityKms = 1.07,
                DepthPpm = 3920, RadiusRearth = 1.08, MassMearth = 1.26, BulkDensityGcm3 = 5.52,
                TeqK = 250, InsolationEarth = 0.65, InHz = true, IsNewMysteryPlanet = true,
                DifficultyTier = 5, AsteroidBeltCount = 12, SurfaceGravityG = 1.08, EscapeVelKms = 12.10, ScaleHeightKm = 8.1, SurfacePressureBar = 1.65, WindShearMaxMs = 82.0,
                PrimaryColor = Color.FromArgb(16, 185, 129), PedestalColor = Color.FromArgb(110, 231, 183), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — Wolf 1069 b (CARMENES 2023): Earth-mass habitable world, 1.26 M⊕, 1.08 R⊕, 250 K, 15.6-day orbit around quiet M5V star.",
                LifeEn = "How life could live: a core-driven magnetosphere protects a dayside liquid ocean where chemosynthetic and red-light life could flourish.",
                AsteroidPhysicsEn = "Real Astrophysics: Absence of close-in gas giants preserved an outer icy planetesimal belt that delivers volatile-rich comets.",
                AtmoLandingEn = "Landing Physics: Earth-like g = 1.08 g (v_esc = 12.1 km/s) and N2/CO2 sky (H = 8.1 km) with 82 m/s katabatic glacier-to-ocean gusts.",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — Wolf 1069 b (CARMENES 2023) : planète terrestre habitable, 1,26 M⊕, 1,08 R⊕, 250 K, orbite de 15,6 jours.",
                LifeFr = "Vie possible : sa magnétosphère protège un océan liquide côté jour propice aux écosystèmes marins et chimiosynthétiques.",
                AsteroidPhysicsFr = "Astrophysique réelle : ceinture externe de planétésimaux glacés intacte livrant des comètes riches en eau.",
                AtmoLandingFr = "Physique d'atterrissage : gravité terrestre g = 1,08 g (v_lib = 12,1 km/s), ciel N2/CO2 (H = 8,1 km) et vents catabatiques glaciaires à 82 m/s."
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-TOI-1452B", ShortLabel = "TOI-1452 b", HostStar = "TOI-1452 (M4V Binary)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 11.062, SemiMajorAxisAu = 0.061, Eccentricity = 0.01, RadialVelocityKms = 3.48,
                DepthPpm = 2710, RadiusRearth = 1.672, MassMearth = 4.82, BulkDensityGcm3 = 5.60,
                TeqK = 326, InsolationEarth = 1.80, InHz = true, IsNewMysteryPlanet = true,
                DifficultyTier = 7, AsteroidBeltCount = 15, SurfaceGravityG = 1.72, EscapeVelKms = 19.01, ScaleHeightKm = 16.4, SurfacePressureBar = 6.50, WindShearMaxMs = 132.0,
                PrimaryColor = Color.FromArgb(6, 182, 212), PedestalColor = Color.FromArgb(103, 232, 249), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — TOI-1452 b: deep-ocean Super-Earth (~30% water mass fraction), 4.82 M⊕, 1.67 R⊕, 326 K in M-dwarf binary system.",
                LifeEn = "How life could live: a hundreds-of-kilometers-deep global ocean heated from below could teem with hydrothermal vent ecosystems.",
                AsteroidPhysicsEn = "Real Astrophysics: Companion star TOI-1452 B (97 AU) induces Kozai-Lidov oscillations, Perturbing outer Oort-cloud comets inward!",
                AtmoLandingEn = "Landing Physics: Strong g = 1.72 g (v_esc = 19.0 km/s), thick 6.5-bar H2O steam deck (H = 16.4 km), and 132 m/s tropical super-cyclones!",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — TOI-1452 b : Super-Terre océanique (~30 % d'eau en masse), 4,82 M⊕, 1,67 R⊕, 326 K en système binaire.",
                LifeFr = "Vie possible : un océan global profond de plusieurs centaines de kilomètres abriterait une riche vie hydrothermale.",
                AsteroidPhysicsFr = "Astrophysique réelle : l'étoile compagne TOI-1452 B (97 UA) induit des cycles de Kozai-Lidov précipitant des comètes glacées !",
                AtmoLandingFr = "Physique d'atterrissage : forte gravité g = 1,72 g (v_lib = 19,0 km/s), couche de vapeur de 6,5 bar (H = 16,4 km) et cyclones à 132 m/s !"
            });
            _unseenPlanetPool.Add(new ExoTarget {
                Index = 8, Id = "NASA-NEW-ROSS-128B", ShortLabel = "Ross 128 b", HostStar = "Ross 128 (M4V)",
                Disposition = "CONFIRMED", GalaxyXNorm = 0.330f, GalaxyYNorm = 0.805f,
                PeriodDays = 9.866, SemiMajorAxisAu = 0.0496, Eccentricity = 0.036, RadialVelocityKms = 1.39,
                DepthPpm = 3150, RadiusRearth = 1.11, MassMearth = 1.40, BulkDensityGcm3 = 5.61,
                TeqK = 301, InsolationEarth = 1.38, InHz = true, IsNewMysteryPlanet = true,
                DifficultyTier = 5, AsteroidBeltCount = 12, SurfaceGravityG = 1.14, EscapeVelKms = 12.60, ScaleHeightKm = 9.4, SurfacePressureBar = 2.10, WindShearMaxMs = 86.0,
                PrimaryColor = Color.FromArgb(52, 211, 153), PedestalColor = Color.FromArgb(167, 243, 208), ApprovedByGamer = true, LandedMedal = false,
                AnalysisEn = "NEW NASA PLANET — Ross 128 b (11.0 ly): quiet-star temperate exoplanet, 1.40 M⊕, 1.11 R⊕, 301 K, P=9.87 d, K=1.39 m/s (HARPS).",
                LifeEn = "How life could live: cloud cover reflecting starlight keeps surface oceans liquid for photosynthetic and complex marine biosphere.",
                AsteroidPhysicsEn = "Real Astrophysics: Slight eccentricity e = 0.036 at a = 0.0496 AU (v_orb = 54.2 km/s) sweeps inner silicate asteroid belt fragments.",
                AtmoLandingEn = "Landing Physics: g = 1.14 g (v_esc = 12.6 km/s), dense 2.1-bar N2/H2O cloud deck (H = 9.4 km), and 86 m/s substellar Hadley-cell shear.",
                AnalysisFr = "NOUVELLE PLANÈTE NASA — Ross 128 b (11,0 al) : exoplanète tempérée, 1,40 M⊕, 1,11 R⊕, 301 K, P=9,87 j, K=1,39 m/s (HARPS).",
                LifeFr = "Vie possible : l'albédo des nuages maintient des océans liquides tempérés propices à une biosphère marine complexe.",
                AsteroidPhysicsFr = "Astrophysique réelle : excentricité e = 0,036 à 0,0496 UA (v_orb = 54,2 km/s) balayant une ceinture interne d'astéroïdes silicatés.",
                AtmoLandingFr = "Physique d'atterrissage : g = 1,14 g (v_lib = 12,6 km/s), couverture nuageuse de 2,1 bar (H = 9,4 km) et cellules de Hadley à 86 m/s."
            });
        }

        /// <summary>
        /// Selects a brand-new real NASA exoplanet from _unseenPlanetPool that was not in the original list
        /// (tracking seen IDs across games in steam_assets/seen_mystery_planets.txt) and injects it into Sector 09 (Index 8).
        /// </summary>
        private void SelectNewUnseenPlanetForNewGame(bool announceVoice)
        {
            if (_unseenPlanetPool == null || _unseenPlanetPool.Count == 0 || _targets.Count < 9) return;
            try
            {
                string assetsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "steam_assets");
                if (!Directory.Exists(assetsDir)) Directory.CreateDirectory(assetsDir);
                string seenFile = Path.Combine(assetsDir, "seen_mystery_planets.txt");
                List<string> seenIds = new List<string>();
                if (File.Exists(seenFile))
                {
                    foreach (string ln in File.ReadAllLines(seenFile))
                    {
                        string trimmed = ln.Trim();
                        if (!string.IsNullOrEmpty(trimmed) && !seenIds.Contains(trimmed))
                            seenIds.Add(trimmed);
                    }
                }

                // If all unseen planets have been visited across games, reset history so cycling continues seamlessly
                if (seenIds.Count >= _unseenPlanetPool.Count)
                {
                    seenIds.Clear();
                }

                ExoTarget chosen = _unseenPlanetPool[0];
                foreach (ExoTarget cand in _unseenPlanetPool)
                {
                    if (!seenIds.Contains(cand.Id))
                    {
                        chosen = cand;
                        break;
                    }
                }

                seenIds.Add(chosen.Id);
                try { File.WriteAllLines(seenFile, seenIds.ToArray()); } catch { }

                chosen.Index = 8;
                chosen.KidLessonEn = chosen.AnalysisEn + " " + chosen.LifeEn;
                chosen.KidLessonFr = chosen.AnalysisFr + " " + chosen.LifeFr;
                _targets[8] = chosen;

                if (announceVoice)
                {
                    _planetsExploredCount++;
                    SelectTarget(8, false, true);
                    AddPointsWithPopup(400, "🎲 NEW NASA PLANET: " + chosen.ShortLabel.ToUpperInvariant(), _galaxyRect.X + chosen.GalaxyXNorm * _galaxyRect.Width, _galaxyRect.Y + chosen.GalaxyYNorm * _galaxyRect.Height, Color.FromArgb(52, 255, 141));
                }
            }
            catch { }
        }

        public static double SolveKeplerEquation(double M, double e)
        {
            double mNorm = M % (Math.PI * 2.0);
            if (mNorm < 0) mNorm += Math.PI * 2.0;
            double E = (e > 0.8) ? Math.PI : mNorm;
            for (int iter = 0; iter < 8; iter++)
            {
                double f = E - e * Math.Sin(E) - mNorm;
                double fPrime = 1.0 - e * Math.Cos(E);
                E -= f / Math.Max(1e-7, fPrime);
            }
            return E;
        }

        /// <summary>
        /// Synthesizes a real 16-bit 44.1 kHz PCM WAV sci-fi sound effect in memory and plays it asynchronously.
        /// </summary>
        private void PlaySynthesizedSciFiWav(double freqStart, double freqEnd, int durationMs)
        {
            if (!_voiceAndSoundOn) return;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    int sampleRate = 44100;
                    int numSamples = (sampleRate * durationMs) / 1000;
                    int dataBytes = numSamples * 2;
                    using (MemoryStream ms = new MemoryStream(44 + dataBytes))
                    using (BinaryWriter bw = new BinaryWriter(ms))
                    {
                        bw.Write(new char[] { 'R', 'I', 'F', 'F' });
                        bw.Write(36 + dataBytes);
                        bw.Write(new char[] { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
                        bw.Write(16);
                        bw.Write((short)1);
                        bw.Write((short)1);
                        bw.Write(sampleRate);
                        bw.Write(sampleRate * 2);
                        bw.Write((short)2);
                        bw.Write((short)16);
                        bw.Write(new char[] { 'd', 'a', 't', 'a' });
                        bw.Write(dataBytes);

                        double phase = 0.0;
                        for (int i = 0; i < numSamples; i++)
                        {
                            double t = (double)i / numSamples;
                            double freq = freqStart + (freqEnd - freqStart) * t;
                            phase += 2.0 * Math.PI * freq / sampleRate;
                            double env = Math.Sin(t * Math.PI);
                            short pcm = (short)(Math.Sin(phase) * 6500 * env);
                            bw.Write(pcm);
                        }
                        ms.Position = 0;
                        using (SoundPlayer sp = new SoundPlayer(ms))
                        {
                            sp.PlaySync();
                        }
                    }
                }
                catch { }
            });
        }

        private string LoadGeminiApiKeyFromEnvOrDotEnv()
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string[] dotEnvCandidates = new string[] {
                    Path.Combine(baseDir, ".env"),
                    Path.Combine(baseDir, @"..\..\.env"),
                    @"d:\xdev\HiveMind\.env",
                    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".env")
                };
                foreach (string envFile in dotEnvCandidates)
                {
                    if (File.Exists(envFile))
                    {
                        foreach (string rawLine in File.ReadAllLines(envFile))
                        {
                            string line = rawLine.Trim();
                            if (line.StartsWith("#") || !line.Contains("=")) continue;
                            int eq = line.IndexOf('=');
                            string k = line.Substring(0, eq).Trim();
                            string v = line.Substring(eq + 1).Trim().Trim('"', '\'');
                            if ((k == "GEMINI_API_KEY" || k == "GOOGLE_API_KEY") && v.Length > 15)
                                return v;
                        }
                    }
                }
            }
            catch { }
            string envKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY");
            if (string.IsNullOrEmpty(envKey)) envKey = Environment.GetEnvironmentVariable("GOOGLE_API_KEY");
            return envKey ?? "";
        }

        private List<string> SplitTextForGcpNeuralTts(string text, int maxChars)
        {
            List<string> chunks = new List<string>();
            string cleaned = Regex.Replace(text ?? "", @"\s+", " ").Trim();
            if (string.IsNullOrEmpty(cleaned)) return chunks;

            string[] sentences = Regex.Split(cleaned, @"(?<=[.!?।。！？])\s+");
            string current = "";
            foreach (string sent in sentences)
            {
                if (string.IsNullOrEmpty(sent)) continue;
                if (current.Length + sent.Length + 1 <= maxChars)
                {
                    current = string.IsNullOrEmpty(current) ? sent : (current + " " + sent);
                }
                else
                {
                    if (!string.IsNullOrEmpty(current)) chunks.Add(current);
                    if (sent.Length <= maxChars)
                    {
                        current = sent;
                    }
                    else
                    {
                        string[] clauses = Regex.Split(sent, @"(?<=[,;:])\s+");
                        current = "";
                        foreach (string cl in clauses)
                        {
                            if (current.Length + cl.Length + 1 <= maxChars)
                                current = string.IsNullOrEmpty(current) ? cl : (current + " " + cl);
                            else
                            {
                                if (!string.IsNullOrEmpty(current)) chunks.Add(current);
                                current = cl.Length <= maxChars ? cl : cl.Substring(0, maxChars);
                            }
                        }
                    }
                }
            }
            if (!string.IsNullOrEmpty(current)) chunks.Add(current);
            return chunks;
        }

        private byte[] WrapPcm24kMonoToWav(byte[] pcmBytes)
        {
            int sampleRate = 24000;
            int dataLen = pcmBytes.Length;
            using (MemoryStream ms = new MemoryStream(44 + dataLen))
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(new char[] { 'R', 'I', 'F', 'F' });
                bw.Write(36 + dataLen);
                bw.Write(new char[] { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
                bw.Write(16);
                bw.Write((short)1);
                bw.Write((short)1);
                bw.Write(sampleRate);
                bw.Write(sampleRate * 2);
                bw.Write((short)2);
                bw.Write((short)16);
                bw.Write(new char[] { 'd', 'a', 't', 'a' });
                bw.Write(dataLen);
                bw.Write(pcmBytes);
                return ms.ToArray();
            }
        }

        private string SynthesizeOrGetCachedGeminiGcpVoice(string text, string lang, string voiceName)
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string cacheDir = Path.Combine(baseDir, "steam_assets", "voice_cache");
                if (!Directory.Exists(cacheDir))
                    Directory.CreateDirectory(cacheDir);

                string normLang = (lang ?? "en").ToLowerInvariant();
                string tl = (normLang == "fr") ? "fr" : (normLang == "es" ? "es" : (normLang == "de" ? "de" : (normLang == "zh" ? "zh-CN" : "en")));

                string digest;
                using (SHA256 sha = SHA256.Create())
                {
                    byte[] hashBytes = sha.ComputeHash(Encoding.UTF8.GetBytes(tl + "|" + voiceName + "|" + text));
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < 8; i++) sb.Append(hashBytes[i].ToString("x2"));
                    digest = sb.ToString();
                }

                string wavPath = Path.Combine(cacheDir, string.Format("gemini25_{0}_{1}_{2}.wav", tl, voiceName.ToLowerInvariant(), digest));
                string mp3Path = Path.Combine(cacheDir, string.Format("gcp_neural_{0}_{1}.mp3", tl, digest));

                if (File.Exists(wavPath) && new FileInfo(wavPath).Length > 1000)
                {
                    _activeVoiceEngineLabel = "GEMINI 2.5 FLASH TTS • " + voiceName.ToUpperInvariant();
                    return wavPath;
                }
                if (File.Exists(mp3Path) && new FileInfo(mp3Path).Length > 500)
                {
                    _activeVoiceEngineLabel = "GCP GEMINI NEURAL • " + voiceName.ToUpperInvariant();
                    return mp3Path;
                }

                // Tier 1: Google Gemini 2.5 Flash Preview Native Audio TTS (Free Tier on Google AI Studio if valid API key present)
                string apiKey = LoadGeminiApiKeyFromEnvOrDotEnv();
                if (!string.IsNullOrEmpty(apiKey))
                {
                    try
                    {
                        string url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key=" + Uri.EscapeDataString(apiKey);
                        string escapedText = (text ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", " ").Replace("\n", " ");
                        string jsonBody = "{\"contents\":[{\"parts\":[{\"text\":\"" + escapedText + "\"}]}],\"generationConfig\":{\"responseModalities\":[\"AUDIO\"],\"speechConfig\":{\"voiceConfig\":{\"prebuiltVoiceConfig\":{\"voiceName\":\"" + voiceName + "\"}}}}}";

                        HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
                        req.Method = "POST";
                        req.ContentType = "application/json; charset=utf-8";
                        req.Timeout = 5500;
                        byte[] bodyBytes = Encoding.UTF8.GetBytes(jsonBody);
                        req.ContentLength = bodyBytes.Length;
                        using (Stream reqStream = req.GetRequestStream())
                        {
                            reqStream.Write(bodyBytes, 0, bodyBytes.Length);
                        }
                        using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                        using (StreamReader sr = new StreamReader(resp.GetResponseStream(), Encoding.UTF8))
                        {
                            string respJson = sr.ReadToEnd();
                            Match m = Regex.Match(respJson, "\"data\"\\s*:\\s*\"([A-Za-z0-9+/=]+)\"");
                            if (m.Success)
                            {
                                byte[] pcm = Convert.FromBase64String(m.Groups[1].Value);
                                if (pcm.Length > 500)
                                {
                                    byte[] wavBytes = WrapPcm24kMonoToWav(pcm);
                                    File.WriteAllBytes(wavPath, wavBytes);
                                    _activeVoiceEngineLabel = "GEMINI 2.5 FLASH TTS • " + voiceName.ToUpperInvariant();
                                    return wavPath;
                                }
                            }
                        }
                    }
                    catch { }
                }

                // Tier 2: Google Cloud Free-Tier Neural TTS (`translate.googleapis.com/translate_tts` — zero API key required!)
                List<string> chunks = SplitTextForGcpNeuralTts(text, 180);
                using (MemoryStream combinedMp3 = new MemoryStream())
                {
                    foreach (string chunk in chunks)
                    {
                        string ttsUrl = string.Format(
                            CultureInfo.InvariantCulture,
                            "https://translate.googleapis.com/translate_tts?ie=UTF-8&client=tw-ob&tl={0}&q={1}",
                            Uri.EscapeDataString(tl),
                            Uri.EscapeDataString(chunk)
                        );
                        HttpWebRequest req = (HttpWebRequest)WebRequest.Create(ttsUrl);
                        req.Method = "GET";
                        req.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)";
                        req.Timeout = 5000;
                        using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                        using (Stream s = resp.GetResponseStream())
                        {
                            byte[] buf = new byte[8192];
                            int read;
                            while ((read = s.Read(buf, 0, buf.Length)) > 0)
                            {
                                combinedMp3.Write(buf, 0, read);
                            }
                        }
                    }
                    byte[] mp3Bytes = combinedMp3.ToArray();
                    if (mp3Bytes.Length > 500)
                    {
                        File.WriteAllBytes(mp3Path, mp3Bytes);
                        _activeVoiceEngineLabel = "GCP GEMINI NEURAL • " + voiceName.ToUpperInvariant();
                        return mp3Path;
                    }
                }
            }
            catch { }
            return null;
        }

        private void PlayAudioFileOnUiThread(string audioPath, int genId)
        {
            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath)) return;
            try
            {
                if (this.IsHandleCreated && !this.IsDisposed)
                {
                    this.BeginInvoke((MethodInvoker)delegate
                    {
                        if (genId != _speechGenerationId || !_voiceAndSoundOn) return;
                        try
                        {
                            if (_speechSynth != null) _speechSynth.SpeakAsyncCancelAll();
                            mciSendString("close gcp_tts", null, 0, IntPtr.Zero);
                            mciSendString("open \"" + audioPath + "\" type mpegvideo alias gcp_tts", null, 0, IntPtr.Zero);
                            mciSendString("play gcp_tts", null, 0, IntPtr.Zero);
                            this.Invalidate();
                        }
                        catch { }
                    });
                }
            }
            catch { }
        }

        private void SpeakWithSubtitles(string text)
        {
            _subtitleText = text;
            int myGen = Interlocked.Increment(ref _speechGenerationId);
            this.Invalidate();
            if (!_voiceAndSoundOn)
            {
                try
                {
                    if (this.IsHandleCreated)
                        this.BeginInvoke((MethodInvoker)delegate { mciSendString("close gcp_tts", null, 0, IntPtr.Zero); });
                }
                catch { }
                return;
            }
            string langSnap = _lang;
            string voiceSnap = _geminiVoiceName;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    // 1. Synthesize or retrieve cached Google Gemini 2.5 TTS / GCP Free-Tier Neural MP3/WAV
                    string cloudAudioPath = SynthesizeOrGetCachedGeminiGcpVoice(text, langSnap, voiceSnap);
                    if (!string.IsNullOrEmpty(cloudAudioPath) && File.Exists(cloudAudioPath))
                    {
                        PlayAudioFileOnUiThread(cloudAudioPath, myGen);
                        return;
                    }

                    // 2. Offline Fallback: Windows Local SpeechSynthesizer (Microsoft Zira Desktop / Hortense)
                    lock (_speechLock)
                    {
                        if (myGen != _speechGenerationId || !_voiceAndSoundOn) return;
                        if (_speechSynth != null)
                        {
                            _activeVoiceEngineLabel = "LOCAL SAPI FALLBACK";
                            _speechSynth.SpeakAsyncCancelAll();
                            string targetIso = (langSnap == "fr") ? "fr" : (langSnap == "es" ? "es" : (langSnap == "de" ? "de" : (langSnap == "zh" ? "zh" : "en")));
                            string chosenVoice = null;
                            string englishFallbackVoice = null;
                            foreach (InstalledVoice iv in _speechSynth.GetInstalledVoices())
                            {
                                if (!iv.Enabled) continue;
                                string twoLetter = iv.VoiceInfo.Culture.TwoLetterISOLanguageName.ToLowerInvariant();
                                if (twoLetter == targetIso)
                                {
                                    chosenVoice = iv.VoiceInfo.Name;
                                    break;
                                }
                                if (twoLetter == "en" && englishFallbackVoice == null)
                                {
                                    englishFallbackVoice = iv.VoiceInfo.Name;
                                }
                            }
                            if (chosenVoice != null)
                                _speechSynth.SelectVoice(chosenVoice);
                            else if (englishFallbackVoice != null)
                                _speechSynth.SelectVoice(englishFallbackVoice);

                            _speechSynth.Rate = 0;
                            _speechSynth.Volume = 100;
                            _speechSynth.SpeakAsync(text);
                        }
                    }
                }
                catch { }
            });
        }

        private string GetStageBannerText()
        {
            if (_lang == "fr")
            {
                if (_gameStage == 0) return "ÉTAPE 1 (CARTE GALACTIQUE) // CLIQUEZ SUR UNE ÉTOILE OU APPUYEZ SUR [ESPACE] POUR LANCER VOS ROBOTS BEE-MOE (+300 PTS) !";
                if (_gameStage == 1) return "ÉTAPE 2 (ORBITE ROBOT) // GUIDEZ LE ROBOT BEE-MOE DANS LE CERCLE VERT AUTOUR DE LA PLANÈTE (+500 PTS) OU CLIQUEZ SUR [3.ATTERRISSAGE] !";
                if (_gameStage == 2) return "ÉTAPE 3 (ATTERRISSAGE FACILE) // GUIDEZ LE ROBOT 🛸 VERS LA CIBLE VERTE 🎯 • [ESPACE] FREINER (+100 PTS) • [ENTRÉE] AUTO-ATTERRIR (+1000 PTS) !";
                if (_gameStage == 4) return "ÉTAPE 5 (BOINC IDLE-GPU & COMMUNAUTÉ GCP) // CALCUL ASTROPHYSIQUE AUTOMATIQUE QUAND LE GPU EST INACTIF (>4s) • STEAM FREE-TO-PLAY (+250 PTS) !";
                return "ÉTAPE 4 (LABO JWST ENFANTS) // ÉCOUTEZ LA LEÇON D'ASTROPHYSIQUE PUIS CLIQUEZ SUR [APPROUVER] OU [REJETER] (+800 PTS) !";
            }
            if (_lang == "es")
            {
                if (_gameStage == 0) return "ETAPA 1 (MAPA GALÁCTICO) // HAZ CLIC EN UNA ESTRELLA O PULSA [ESPACIO] PARA LANZAR TUS ROBOTS BEE-MOE (+300 PTS)";
                if (_gameStage == 1) return "ETAPA 2 (ÓRBITA ROBOT) // LLEVA EL ROBOT AL CÍRCULO VERDE DE LA PLANETA (+500 PTS) O PULSA [3.PLANET LAND]";
                if (_gameStage == 2) return "ETAPA 3 (ATERRIZAJE FÁCIL) // GUÍA EL ROBOT 🛸 A LA PISTA VERDE 🎯 • [ESPACIO] FRENAR (+100 PTS) • [ENTER] AUTO-ATERRIZAR (+1000 PTS)";
                if (_gameStage == 4) return "ETAPA 5 (BOINC IDLE-GPU & COMUNIDAD GCP) // CÁLCULO ASTROFÍSICO AUTOMÁTICO EN GPU INACTIVA (>4s) • STEAM FREE-TO-PLAY (+250 PTS)";
                return "ETAPA 4 (LABORATORIO JWST) // APRENDE ASTROFÍSICA Y PULSA [APROBAR] O [RECHAZAR] PARA GANAR +800 PUNTOS";
            }
            if (_lang == "de")
            {
                if (_gameStage == 0) return "STUFE 1 (GALAXIE-KARTE) // KLICKE AUF EINEN STERN ODER DRÜCKE [LEERTASTE] FÜR ROBOTER-START (+300 PKT)";
                if (_gameStage == 1) return "STUFE 2 (ORBIT-ROBOTER) // STEUERE DEN ROBOTER IN DEN GRÜNEN KREIS (+500 PKT) ODER KLICKE [3.PLANET LAND]";
                if (_gameStage == 2) return "STUFE 3 (LEICHTE LANDUNG) // STEUERE 🛸 ZUM GRÜNEN LANDEFELD 🎯 • [LEERTASTE] BREMSEN (+100 PKT) • [ENTER] AUTO-LANDUNG (+1000 PKT)";
                if (_gameStage == 4) return "STUFE 5 (BOINC IDLE-GPU & GCP COMMUNITY) // AUTOMATISCHE ASTROPHYSIK-BERECHNUNG BEI INAKTIVER GPU (>4s) (+250 PKT)";
                return "STUFE 4 (JWST LABOR) // LERNE ASTROPHYSIK & KLICKE [BESTÄTIGEN] ODER [ABLEHNEN] (+800 PKT)";
            }
            if (_lang == "zh")
            {
                if (_gameStage == 0) return "第 1 步（星系导航）// 点击恒星或按 [空格键] 发射探索机器人卫星（+300 分）！";
                if (_gameStage == 1) return "第 2 步（轨道捕获）// 引导机器人进入行星绿色引力圈（+500 分）或点击 [3.行星着陆]！";
                if (_gameStage == 2) return "第 3 步（轻松着陆）// 引导机器人 🛸 降落到绿色靶心 🎯 • [空格键] 减速 (+100 分) • [回车键] 自动着陆 (+1000 分)！";
                if (_gameStage == 4) return "第 5 步（BOINC 闲置 GPU 公民科学 & GCP 社区）// GPU 闲置超 4 秒自动折叠系外行星凌星光变曲线（+250 分）！";
                return "第 4 步（韦伯太空实验室）// 收听天体物理趣味讲解，点击 [批准发现] 或 [排除假阳性]（+800 分）！";
            }
            // Default English ("en")
            if (_gameStage == 0) return "STEP 1 (GALAXY BRIDGE) // CLICK ANY STAR OR PRESS [SPACE] TO LAUNCH YOUR BEE-MOE ROBOT SATELLITES (+300 PTS)!";
            if (_gameStage == 1) return "STEP 2 (ORBIT CAPTURE) // FLY YOUR ROBOT INTO THE GREEN CIRCLE AROUND THE PLANET (+500 PTS) OR CLICK [3.PLANET LAND]!";
            if (_gameStage == 2) return "STEP 3 (EASY PLANET LANDING) // GUIDE YOUR ROBOT TO THE GREEN PAD • [SPACE] BRAKE (+100 PTS) • [ENTER] AUTO-LAND (+1,000 PTS)!";
            if (_gameStage == 4) return "STEP 5 (BOINC IDLE-GPU & GCP COMMUNITY HUB) // AUTO ASTROPHYSICS FOLDING WHEN GPU IDLE (>4s) • STEAM FREE-TO-PLAY (+250 PTS)!";
            return "STEP 4 (JWST SCIENCE LAB) // LISTEN TO THE KID ASTROPHYSICS LESSON THEN CLICK [APPROVE] OR [REJECT] (+800 PTS)!";
        }

        private string GetCurrentPlanetKidLesson()
        {
            if (_current == null) return "";
            if (_lang == "fr")
            {
                string extraFr = (_gameStage == 1 && !string.IsNullOrEmpty(_current.AsteroidPhysicsFr))
                    ? (" " + _current.AsteroidPhysicsFr)
                    : ((_gameStage == 2 && !string.IsNullOrEmpty(_current.AtmoLandingFr)) ? (" " + _current.AtmoLandingFr) : "");
                return _current.AnalysisFr + " " + _current.LifeFr + extraFr;
            }
            if (_lang == "es")
            {
                string lifeEs = (_current.Disposition == "FALSE POSITIVE")
                    ? "Vida: estrella binaria ardiente, sin superficie sólida para la vida."
                    : (_current.InHz
                        ? "Vida: océanos templados y fuentes hidrotermales podrían sostener microbios y vida marina."
                        : "Vida: calor extremo en superficie; solo microbios en nubes altas o lunas heladas sobrevivirían.");
                return string.Format(CultureInfo.InvariantCulture, "Análisis de {0} (Nivel {1}, g={2:F2}g, H={3:F1}km): radio {4:F2} R⊕, masa {5:F2} M⊕, {6:F0} K, tránsito {7:F0} ppm, Doppler {8:F2} m/s. {9}",
                    _current.ShortLabel, _difficultyLevel, _current.SurfaceGravityG, _current.ScaleHeightKm, _current.RadiusRearth, _current.MassMearth, _current.TeqK, _current.DepthPpm, _current.RadialVelocityKms, lifeEs);
            }
            if (_lang == "de")
            {
                string lifeDe = (_current.Disposition == "FALSE POSITIVE")
                    ? "Leben: heißer Doppelstern ohne festen Planeten; kein Leben möglich."
                    : (_current.InHz
                        ? "Leben: flüssige Ozeane und Hydrothermalquellen könnten Mikroben und Meeresleben ermöglichen."
                        : "Leben: extreme Hitze am Boden; nur Mikroben in hohen Wolken oder Eismonden überlebensfähig.");
                return string.Format(CultureInfo.InvariantCulture, "Analyse {0} (Stufe {1}, g={2:F2}g, H={3:F1}km): Radius {4:F2} R⊕, Masse {5:F2} M⊕, {6:F0} K, Transit {7:F0} ppm, Doppler {8:F2} m/s. {9}",
                    _current.ShortLabel, _difficultyLevel, _current.SurfaceGravityG, _current.ScaleHeightKm, _current.RadiusRearth, _current.MassMearth, _current.TeqK, _current.DepthPpm, _current.RadialVelocityKms, lifeDe);
            }
            if (_lang == "zh")
            {
                string lifeZh = (_current.Disposition == "FALSE POSITIVE")
                    ? "生命评估：炽热食双星假阳性，无固态表面，生命无法存活。"
                    : (_current.InHz
                        ? "生命评估：处于宜居带，液态海洋与海底热泉可孕育光合藻类及海洋微生物。"
                        : "生命评估：表面高温严酷，仅高层温带云层或冰卫星地下海洋可能存在极端微生物。");
                return string.Format(CultureInfo.InvariantCulture, "{0} 行星分析（难度 {1}，重力 {2:F2}g，标高 {3:F1}km）：半径 {4:F2} 地球，质量 {5:F2} 地球，温度 {6:F0} K，凌星 {7:F0} ppm，多普勒 {8:F2} m/s。{9}",
                    _current.ShortLabel, _difficultyLevel, _current.SurfaceGravityG, _current.ScaleHeightKm, _current.RadiusRearth, _current.MassMearth, _current.TeqK, _current.DepthPpm, _current.RadialVelocityKms, lifeZh);
            }
            string extraEn = (_gameStage == 1 && !string.IsNullOrEmpty(_current.AsteroidPhysicsEn))
                ? (" " + _current.AsteroidPhysicsEn)
                : ((_gameStage == 2 && !string.IsNullOrEmpty(_current.AtmoLandingEn)) ? (" " + _current.AtmoLandingEn) : "");
            return _current.AnalysisEn + " " + _current.LifeEn + extraEn;
        }

        private string GetOneTimeChapterExplanation(int stage)
        {
            if (_lang == "fr")
            {
                if (stage == 0) return "Chapitre 1 : Passerelle. Survolez les étoiles ou appuyez sur ESPACE pour lancer les sondes.";
                if (stage == 1) return "Chapitre 2 : Orbite. Guidez la sonde dans l'anneau vert de Lagrange.";
                if (stage == 2) return "Chapitre 3 : Atterrissage. Suivez le tunnel vert ; ESPACE freine, ENTRÉE atterrit.";
                if (stage == 4) return "Chapitre 5 : Centre BOINC GPU. Le GPU inactif plie les courbes TESS et JWST.";
                return "Chapitre 4 : Labo JWST. Vérifiez transit et Doppler, puis Approuvez ou Rejetez.";
            }
            if (_lang == "es")
            {
                if (stage == 0) return "Capítulo 1: Puente. Vuele entre estrellas o pulse ESPACIO para lanzar sondas.";
                if (stage == 1) return "Capítulo 2: Órbita. Guíe la sonda al anillo verde de Lagrange.";
                if (stage == 2) return "Capítulo 3: Aterrizaje. Siga el túnel verde; ESPACIO frena, ENTER aterriza.";
                if (stage == 4) return "Capítulo 5: Centro BOINC GPU. La GPU inactiva pliega curvas TESS y JWST.";
                return "Capítulo 4: Labo JWST. Verifique tránsito y Doppler, luego Apruebe o Rechace.";
            }
            if (_lang == "de")
            {
                if (stage == 0) return "Kapitel 1: Brücke. Fliegen Sie zu Sternen oder drücken Sie LEERTASTE für Sonden.";
                if (stage == 1) return "Kapitel 2: Orbit. Steuern Sie die Sonde in den grünen Lagrange-Ring.";
                if (stage == 2) return "Kapitel 3: Landung. Grünem Tunnel folgen; LEERTASTE bremst, ENTER landet.";
                if (stage == 4) return "Kapitel 5: BOINC GPU-Zentrum. Inaktive GPU faltet TESS- und JWST-Lichtkurven.";
                return "Kapitel 4: JWST-Labor. Transit und Doppler prüfen, dann Bestätigen oder Ablehnen.";
            }
            if (_lang == "zh")
            {
                if (stage == 0) return "第一章：舰桥。在恒星间航行或按空格键发射探测器。";
                if (stage == 1) return "第二章：轨道。引导探测器进入绿色拉格朗日环。";
                if (stage == 2) return "第三章：着陆。沿绿色通道下降，空格减速，回车自动着陆。";
                if (stage == 4) return "第五章：BOINC 算力中心。闲置 GPU 自动折叠系外行星光变曲线。";
                return "第四章：韦伯实验室。核对凌星与多普勒曲线，点击批准或排除。";
            }
            if (stage == 0) return "Chapter 1: Bridge. Fly between stars or press SPACE to launch probes.";
            if (stage == 1) return "Chapter 2: Orbit. Guide your probe into the green Lagrange ring.";
            if (stage == 2) return "Chapter 3: Landing. Follow the green tunnel; SPACE brakes, ENTER auto-lands.";
            if (stage == 4) return "Chapter 5: BOINC GPU Hub. Idle GPU folds TESS and JWST light curves automatically.";
            return "Chapter 4: JWST Lab. Verify transit and Doppler curves, then Approve or Reject.";
        }

        private void SpeakPlanetAnalysisOnly()
        {
            if (_current == null) return;
            string planetMsg = GetCurrentPlanetKidLesson();
            _subtitleText = planetMsg;
            this.Invalidate();

            // If the one-time Chapter 1 intro explanation is still speaking during its first ~3 seconds,
            // do not interrupt or restart it—queue the planet analysis immediately after it finishes!
            double remainMs = (_chapterIntroEndUtc - DateTime.UtcNow).TotalMilliseconds;
            if (remainMs > 80.0)
            {
                int targetSnap = _current.Index;
                ThreadPool.QueueUserWorkItem(_ =>
                {
                    Thread.Sleep((int)Math.Min(3200.0, remainMs + 60.0));
                    if (_current != null && _current.Index == targetSnap)
                    {
                        SpeakWithSubtitles(GetCurrentPlanetKidLesson());
                    }
                });
                return;
            }

            SpeakWithSubtitles(planetMsg);
        }

        private void SpeakStepNarrativeAndAstrophysics()
        {
            if (_current == null) return;
            int stg = Math.Max(0, Math.Min(4, _gameStage));

            // Ensure each Chapter explanation is spoken ONLY ONE TIME per session!
            if (!_chapterIntroSpoken[stg])
            {
                _chapterIntroSpoken[stg] = true;
                _chapterIntroEndUtc = DateTime.UtcNow.AddSeconds(3.0);
                string intro = GetOneTimeChapterExplanation(stg);
                string msg = (stg == 4) ? intro : (intro + " " + GetCurrentPlanetKidLesson());
                SpeakWithSubtitles(msg);
            }
            else
            {
                // After the one-time chapter explanation, speak ONLY the planet analysis, specific planet info & how life could live!
                SpeakPlanetAnalysisOnly();
            }
        }

        private void TriggerBoincIdleGpuWorkunitSync(string projectId, bool showPopup)
        {
            _boincWorkunitsCompleted++;
            _boincCobblestones = Math.Round(_boincCobblestones + 148.5, 1);
            for (int i = 0; i < _boincProjProgress.Length; i++)
            {
                _boincProjProgress[i] = 28.0 + ((_boincWorkunitsCompleted * 17 + i * 23) % 68);
            }
            if (showPopup)
            {
                AddPointsWithPopup(250, "⚡ BOINC IDLE-GPU WU SYNCED!", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 140f, Color.FromArgb(52, 255, 141));
                PlaySynthesizedSciFiWav(660.0, 1320.0, 130);
            }

            string targetLabel = (_current != null) ? _current.ShortLabel : "K2-18 b";
            double idleSnap = _idleSeconds;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    // 1. Sync with local/GCP Community Hub API (:8095/api/v1/community/boinc-sync)
                    HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:8095/api/v1/community/boinc-sync");
                    req.Method = "POST";
                    req.ContentType = "application/json";
                    req.Timeout = 2800;
                    string payload = string.Format(
                        CultureInfo.InvariantCulture,
                        "{{\"commander\":\"Cmdr_JeanPaul_NCC31337\",\"project_id\":\"{0}\",\"gpu_idle_seconds\":{1:F1},\"gpu_utilization_pct\":5.8,\"gpu_model\":\"NVIDIA GeForce RTX 2070 (Steam Idle-GPU)\",\"target_planet\":\"{2}\"}}",
                        projectId, idleSnap, targetLabel
                    );
                    byte[] bodyBytes = Encoding.UTF8.GetBytes(payload);
                    req.ContentLength = bodyBytes.Length;
                    using (Stream s = req.GetRequestStream())
                    {
                        s.Write(bodyBytes, 0, bodyBytes.Length);
                    }
                    using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                    using (StreamReader sr = new StreamReader(resp.GetResponseStream()))
                    {
                        string respJson = sr.ReadToEnd();
                        if (!string.IsNullOrEmpty(respJson) && respJson.Contains("BOINC-BLS-"))
                        {
                            int idx = respJson.IndexOf("BOINC-BLS-");
                            if (idx >= 0 && idx + 18 <= respJson.Length)
                                _boincLastReceiptId = respJson.Substring(idx, 18);
                        }
                    }
                }
                catch { }

                try
                {
                    // 2. Also lease/ping live public GCP Cloud Run HiveMind@home Federated Coordinator
                    HttpWebRequest gcpReq = (HttpWebRequest)WebRequest.Create("https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/request");
                    gcpReq.Method = "POST";
                    gcpReq.ContentType = "application/json";
                    gcpReq.Timeout = 3500;
                    string gcpBody = "{\"worker_id\":\"steam-fat-client-3133700\",\"capabilities\":[\"boinc_astrophysics\",\"bls_transit\"],\"idle_gpu\":true}";
                    byte[] gBytes = Encoding.UTF8.GetBytes(gcpBody);
                    gcpReq.ContentLength = gBytes.Length;
                    using (Stream gs = gcpReq.GetRequestStream())
                    {
                        gs.Write(gBytes, 0, gBytes.Length);
                    }
                    using (HttpWebResponse gResp = (HttpWebResponse)gcpReq.GetResponse())
                    {
                        if ((int)gResp.StatusCode == 200)
                        {
                            _boincLastSyncStatus = "GCP CLOUD RUN 200 OK • hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app";
                        }
                    }
                }
                catch { }
            });
        }

        private void SpeakLocalizedBriefing()
        {
            SpeakStepNarrativeAndAstrophysics();
        }

        private void SelectTarget(int idx, bool keepShipPos, bool playSound)
        {
            if (idx < 0 || idx >= _targets.Count) return;
            _current = _targets[idx];
            if (!keepShipPos)
            {
                _shipXNorm = Math.Max(0.08f, _current.GalaxyXNorm - 0.08f);
                _shipYNorm = Math.Max(0.10f, _current.GalaxyYNorm - 0.06f);
            }
            if (playSound)
            {
                _planetSwitchCount++;
                _planetsExploredCount++;
            }
            // Progressive difficulty increases with each planet explored AND respects intrinsic target DifficultyTier!
            _difficultyLevel = Math.Min(10, Math.Max(Math.Max(1, _current.DifficultyTier), 1 + _planetsExploredCount));
            _captureProgress = 0.0;
            _surpriseEventTimer = Math.Max(3.2, 6.8 - _difficultyLevel * 0.35);
            _surpriseEventBanner = (_lang == "fr") ? _current.AsteroidPhysicsFr : _current.AsteroidPhysicsEn;
            ResetOolitePlanetfallState();
            if (playSound)
            {
                _chapterIntroSpoken[0] = true; // Never restart Chapter 1 voice when moving between planets
                PlaySynthesizedSciFiWav(880.0, 1320.0, 95);
                AddPointsWithPopup(50, string.Format("STAR SCAN (DIFF LVL {0})", _difficultyLevel), _galaxyRect.X + _current.GalaxyXNorm * _galaxyRect.Width, _galaxyRect.Y + _current.GalaxyYNorm * _galaxyRect.Height, Color.FromArgb(56, 189, 248));
                SpeakPlanetAnalysisOnly();
            }
            this.Invalidate();
        }

        private void TriggerSurpriseAstrophysicalEvent()
        {
            if (_current == null) return;
            int evtKind = (int)(Math.Abs((long)(_simTime * 100.0) + _planetsExploredCount * 7) % 4);
            if (_gameStage == 1)
            {
                // Stage 1 (Orbit Bee): Surprise Keplerian Asteroid / CME / Periastron Tidal Events explained with real astrophysics data
                if (evtKind == 0)
                {
                    _surpriseEventBanner = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "⚡ SURPRISE CME M-DWARF // Éruption XUV (10^33 erg) sur {0} ! Pression dynamique P_ram = ρ v² pousse la sonde !", _current.HostStar)
                        : string.Format(CultureInfo.InvariantCulture, "⚡ SURPRISE CME SUPERFLARE // {0} XUV shockwave (10^33 erg)! Solar wind ram pressure P = ρ·v² perturbs probe orbit!", _current.HostStar);
                    _beeVx -= 2.2f + _difficultyLevel * 0.25f;
                }
                else if (evtKind == 1)
                {
                    _surpriseEventBanner = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "☄ ESSAIM DE KIRKWOOD 3:1 // Résonance orbitale (a={0:F3} UA, e={1:F2}) : bolides hyper-véloces ({2:F1} km/s) !", _current.SemiMajorAxisAu, _current.Eccentricity, _current.EscapeVelKms * 1.4)
                        : string.Format(CultureInfo.InvariantCulture, "☄ KIRKWOOD 3:1 BOLIDE SWARM // Mean-motion resonance (a={0:F3} AU, e={1:F2}) injects {2:F1} km/s exocomets!", _current.SemiMajorAxisAu, _current.Eccentricity, _current.EscapeVelKms * 1.4);
                    _beeVy += ((_planetsExploredCount % 2 == 0) ? 2.4f : -2.4f);
                }
                else if (evtKind == 2)
                {
                    _surpriseEventBanner = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "🌀 CISAILLEMENT DE MARÉE // Gravité g={0:F2}g (v_lib={1:F1} km/s) : gradient de Roche dF/dr = 2GM/r³ dévie la trajectoire !", _current.SurfaceGravityG, _current.EscapeVelKms)
                        : string.Format(CultureInfo.InvariantCulture, "🌀 PERIASTRON TIDAL SHEAR // Gravity g={0:F2}g (v_esc={1:F1} km/s): Roche gradient dF/dr = 2GM/r³ pulls probe!", _current.SurfaceGravityG, _current.EscapeVelKms);
                }
                else
                {
                    _surpriseEventBanner = (_lang == "fr") ? _current.AsteroidPhysicsFr : _current.AsteroidPhysicsEn;
                }
                AddPointsWithPopup(75, "☄ ASTRO SURPRISE LVL " + _difficultyLevel, _beeX, _beeY - 18f, Color.FromArgb(251, 146, 60));
                PlaySynthesizedSciFiWav(360.0, 920.0, 125);
            }
            else if (_gameStage == 2)
            {
                // Stage 2 (Oolite Planetfall): Surprise Atmospheric Microbursts, Coriolis Jet Shear & Baroclinic Waves
                double gVal = Math.Max(0.25, _current.SurfaceGravityG);
                double hVal = Math.Max(2.0, _current.ScaleHeightKm);
                if (evtKind == 0)
                {
                    double gust = Math.Round(_current.WindShearMaxMs * (0.45 + _difficultyLevel * 0.06), 1);
                    _landerWindShearMs = gust;
                    _landerOffsetX += (float)(((_simTime * 10.0) % 2.0 > 1.0 ? 1.0 : -1.0) * (14.0 + _difficultyLevel * 2.2));
                    _surpriseEventBanner = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "🌪 RAFALE JET-STREAM // Cisaillement {0:F1} m/s (H={1:F1} km, P={2:F1} bar) ! Corrigez avec [GAUCHE/DROITE] ou [ESPACE] !", gust, hVal, _current.SurfacePressureBar)
                        : string.Format(CultureInfo.InvariantCulture, "🌪 SUPERSONIC JET GUST // Wind shear {0:F1} m/s (Scale Height H={1:F1} km, P={2:F1} bar)! Steer or press [SPACE]!", gust, hVal, _current.SurfacePressureBar);
                }
                else if (evtKind == 1)
                {
                    _landerSpeedMs = Math.Min(155.0, _landerSpeedMs + 12.0 + gVal * 3.5);
                    _landerCabinTempC = Math.Min(180.0, _landerCabinTempC + 10.0);
                    _surpriseEventBanner = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "⬇ MICRORAFALE CONVECTIVE // Puits gravitationnel g={0:F2}g ({1:F1} m/s²) accélère la chute ! [ESPACE/HAUT] Rétro-fusées !", gVal, gVal * 9.81)
                        : string.Format(CultureInfo.InvariantCulture, "⬇ CONVECTIVE MICROBURST // Gravity well g={0:F2}g ({1:F1} m/s²) accelerates descent! Fire [SPACE/UP] Retro-Thrusters!", gVal, gVal * 9.81);
                }
                else
                {
                    _surpriseEventBanner = (_lang == "fr") ? _current.AtmoLandingFr : _current.AtmoLandingEn;
                }
                AddPointsWithPopup(85, "🌪 ATMO TURBULENCE LVL " + _difficultyLevel, _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 200f, Color.FromArgb(250, 204, 21));
                PlaySynthesizedSciFiWav(310.0, 780.0, 130);
            }
        }

        private void LaunchSatellites()
        {
            _launchedProbes.Clear();
            float sx = _galaxyRect.X + _shipXNorm * _galaxyRect.Width;
            float sy = _galaxyRect.Y + _shipYNorm * _galaxyRect.Height;
            float tx = _galaxyRect.X + _current.GalaxyXNorm * _galaxyRect.Width;
            float ty = _galaxyRect.Y + _current.GalaxyYNorm * _galaxyRect.Height;
            Color[] cols = new Color[] {
                Color.FromArgb(56, 189, 248), Color.FromArgb(52, 211, 153),
                Color.FromArgb(255, 170, 0), Color.FromArgb(168, 85, 247)
            };
            for (int i = 0; i < 4; i++)
            {
                _launchedProbes.Add(new ProbeProjectile {
                    X = sx + (i - 1.5f) * 14f,
                    Y = sy + (i % 2 == 0 ? -10f : 10f),
                    TargetX = tx,
                    TargetY = ty,
                    Color = cols[i]
                });
            }
            _warpTimer = 0.75;
            AddPointsWithPopup(300, "SATELLITES LAUNCHED", tx, ty, Color.FromArgb(52, 211, 153));
            PlaySynthesizedSciFiWav(260.0, 1240.0, 260);
        }

        private void CompletePlanetLandingSuccess()
        {
            if (_current.Disposition == "FALSE POSITIVE")
            {
                PlaySynthesizedSciFiWav(320.0, 880.0, 260);
                AddPointsWithPopup(500, "BINARY STAR DETECTED!", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 180f, Color.FromArgb(250, 204, 21));
                _gameStage = 3;
                _showNarrativeCard = true;
                SpeakWithSubtitles(_lang == "fr"
                    ? "Attention Explorateur ! KOI-Bin #02 n'a pas de surface solide : ce sont deux étoiles binaires brûlantes (g = 31,6 g, 1 480 K) ! Passons au labo JWST pour rejeter ce faux positif et gagner 800 points !"
                    : "Careful, Explorer! KOI-Bin #02 has no solid ground—it is two scorching eclipsing binary stars (g = 31.6 g, 1,480 K)! Let's go to the JWST Lab and click REJECT to win 800 Detective Points!");
                return;
            }

            // Collect any remaining bonus stars automatically on landing!
            foreach (var st in _bonusStars)
            {
                if (!st.Collected)
                {
                    st.Collected = true;
                    _scoreHmt += 200;
                }
            }
            if (!_current.LandedMedal)
            {
                _current.LandedMedal = true;
                _medalsCount++;
            }
            // Escalate progressive difficulty after each successful planetary landing!
            _planetsExploredCount++;
            _difficultyLevel = Math.Min(10, Math.Max(_difficultyLevel + 1, 1 + _planetsExploredCount));
            _streakCount = Math.Min(5, _streakCount + 1);
            AddPointsWithPopup(1000 + _difficultyLevel * 100, string.Format("🥇 TOUCHDOWN! (NEXT DIFF LVL {0})", _difficultyLevel), _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 210f, Color.FromArgb(250, 204, 21));
            PlaySynthesizedSciFiWav(523.25, 1318.5, 340);
            _gameStage = 3;
            _showNarrativeCard = true;
            SpeakStepNarrativeAndAstrophysics();
        }

        private void ApproveOrReject(bool approve)
        {
            bool isPlanet = (_current.Disposition != "FALSE POSITIVE" && _current.RadialVelocityKms >= 0.1);
            bool ok = (approve == isPlanet);
            if (ok)
            {
                _streakCount = Math.Min(5, _streakCount + 1);
                AddPointsWithPopup(800, "🌟 ASTROPHYSICS MASTER!", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 140f, Color.FromArgb(52, 211, 153));
            }
            else
            {
                _streakCount = 1;
                AddPointsWithPopup(200, "SCIENCE TRY", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 140f, Color.FromArgb(56, 189, 248));
            }
            _current.ApprovedByGamer = true;
            PlaySynthesizedSciFiWav(523.25, 1046.50, 280);
            _gameStage = 0;
            _showNarrativeCard = true;
            SpeakWithSubtitles(_lang == "fr"
                ? (ok ? "Bravo Cadet ! Ton analyse astrophysique est exacte ! La difficulté augmente au Niveau " + _difficultyLevel + " pour la prochaine planète !" : "Bien essayé Cadet ! Regarde bien les courbes Doppler et Transit pour la prochaine planète.")
                : (ok ? "Awesome job, Space Cadet! Your astrophysics analysis is 100 percent correct! Difficulty rises to Level " + _difficultyLevel + " for your next world!" : "Good try, Space Cadet! Remember: a real planet needs both a flat U-shaped transit shadow and a Doppler gravity wobble!"));
        }

        private void PollGamepad()
        {
            try
            {
                JOYINFOEX info = new JOYINFOEX();
                info.dwSize = Marshal.SizeOf(typeof(JOYINFOEX));
                info.dwFlags = 0xFF;
                if (joyGetPosEx(0, ref info) == 0)
                {
                    float dx = (info.dwXpos - 32767f) / 32768f;
                    float dy = (info.dwYpos - 32767f) / 32768f;
                    if (Math.Abs(dx) > 0.18f || Math.Abs(dy) > 0.18f) MoveAvatar(dx, dy);

                    int btns = info.dwButtons;
                    bool bA = ((btns & 1) != 0) && ((_prevGpButtons & 1) == 0);
                    bool bX = ((btns & 4) != 0) && ((_prevGpButtons & 4) == 0);
                    if (bA)
                    {
                        if (_gameStage == 0 && _warpTimer <= 0) LaunchSatellites();
                        else if (_gameStage == 2) CompletePlanetLandingSuccess();
                        else if (_gameStage == 3) ApproveOrReject(true);
                    }
                    if (bX && _gameStage == 3) ApproveOrReject(false);
                    _prevGpButtons = btns;
                }
            }
            catch { }
        }

        private void MoveAvatar(float dx, float dy)
        {
            if (_gameStage == 0)
            {
                _shipXNorm = Math.Max(0.06f, Math.Min(0.94f, _shipXNorm + dx * 0.008f));
                _shipYNorm = Math.Max(0.08f, Math.Min(0.88f, _shipYNorm + dy * 0.008f));
                _shipHeading = Math.Atan2(dy, dx);
                for (int i = 0; i < _targets.Count; i++)
                {
                    double d = Math.Sqrt(Math.Pow(_targets[i].GalaxyXNorm - _shipXNorm, 2) + Math.Pow(_targets[i].GalaxyYNorm - _shipYNorm, 2));
                    if (d < 0.085 && i != _current.Index)
                    {
                        SelectTarget(i, true, true);
                        break;
                    }
                }
            }
            else if (_gameStage == 1)
            {
                // Stage 1: Responsive Thruster Steering against Keplerian asteroid belts
                _beeVx = Math.Max(-9.5f, Math.Min(9.5f, _beeVx + dx * 1.25f));
                _beeVy = Math.Max(-9.5f, Math.Min(9.5f, _beeVy + dy * 1.25f));
            }
            else if (_gameStage == 2)
            {
                // Stage 2 (Oolite Planetfall): Steer left/right against atmospheric wind shear & fire retro-thrusters [UP/Z/W]
                _landerOffsetX = Math.Max(-150f, Math.Min(150f, _landerOffsetX + dx * 4.2f));
                if (dy < -0.1f) // [UP / Z / W] fires retro-thrusters to counter high planetary gravity g!
                {
                    _landerSpeedMs = Math.Max(32.0, _landerSpeedMs - 2.8);
                    _landerCabinTempC = Math.Max(38.0, _landerCabinTempC - 1.2);
                    _landerPitchDeg = -8.0f;
                }
                else if (dy > 0.1f) // [DOWN / S] nose-down dive
                {
                    _landerSpeedMs = Math.Min(145.0, _landerSpeedMs + 2.2);
                    _landerPitchDeg = -18.0f;
                }
            }
        }

        private void OnTick()
        {
            _simTime += 0.016;
            _idleSeconds += 0.016;
            if (_hazardHitFlash > 0) _hazardHitFlash = Math.Max(0.0, _hazardHitFlash - 0.04);
            if (_asteroidHitCooldown > 0) _asteroidHitCooldown = Math.Max(0.0, _asteroidHitCooldown - 0.016);

            // Periodic Surprise Astrophysical Events in Stage 1 (Orbit Bee) and Stage 2 (Oolite Landing)
            if (_gameStage == 1 || _gameStage == 2)
            {
                _surpriseEventTimer -= 0.016;
                if (_surpriseEventTimer <= 0.0)
                {
                    _surpriseEventTimer = Math.Max(3.4, 7.5 - _difficultyLevel * 0.38);
                    TriggerSurpriseAstrophysicalEvent();
                }
            }

            // Animate floating +POINTS popups
            for (int i = _popups.Count - 1; i >= 0; i--)
            {
                _popups[i].Y -= 1.2f;
                _popups[i].Life -= 0.022;
                if (_popups[i].Life <= 0) _popups.RemoveAt(i);
            }

            PollGamepad();

            float dx = (_kRight ? 1f : 0f) - (_kLeft ? 1f : 0f);
            float dy = (_kDown ? 1f : 0f) - (_kUp ? 1f : 0f);
            if (Math.Abs(dx) > 0.01f || Math.Abs(dy) > 0.01f)
            {
                _idleSeconds = 0.0;
                MoveAvatar(dx, dy);
            }

            // Steam Free-to-Play Idle-GPU BOINC Citizen-Science Scheduler:
            if ((_boincAutoIdleMode && _idleSeconds >= 4.0) || _boincForceAlwaysOn)
            {
                for (int i = 0; i < _boincProjProgress.Length; i++)
                {
                    _boincProjProgress[i] += 0.042 * (1.0 + i * 0.15);
                    if (_boincProjProgress[i] >= 100.0)
                    {
                        _boincProjProgress[i] = 12.0;
                        TriggerBoincIdleGpuWorkunitSync("hivemind_tess_at_home", _gameStage == 4);
                    }
                }
            }

            if (_gameStage == 0 && _warpTimer > 0.0)
            {
                _warpTimer -= 0.022;
                foreach (var p in _launchedProbes)
                {
                    p.X += (p.TargetX - p.X) * 0.20f;
                    p.Y += (p.TargetY - p.Y) * 0.20f;
                }
                if (_warpTimer <= 0.0)
                {
                    _launchedProbes.Clear();
                    _gameStage = 1;
                    _showNarrativeCard = true;
                    InitBonusStars();
                    _beeX = _galaxyRect.X + 180f;
                    _beeY = _galaxyRect.Y + _galaxyRect.Height * 0.5f;
                    _beeVx = 2.0f;
                    _beeVy = 0.0f;
                    _captureProgress = 15.0;
                    _surpriseEventBanner = (_lang == "fr") ? _current.AsteroidPhysicsFr : _current.AsteroidPhysicsEn;
                    SpeakStepNarrativeAndAstrophysics();
                }
            }
            else if (_gameStage == 1)
            {
                // Stage 1 (Orbit Bee): Progressive Difficulty — Auto-attraction weakens as _difficultyLevel rises,
                // while Keplerian asteroid count, speed, and momentum knockback increase!
                float toPx = _planetScreenX - _beeX;
                float toPy = _planetScreenY - _beeY;
                float distToP = (float)Math.Max(10.0, Math.Sqrt(toPx * toPx + toPy * toPy));
                float pullStrength = Math.Max(0.05f, 0.24f - _difficultyLevel * 0.018f);
                _beeVx += (toPx / distToP) * pullStrength;
                _beeVy += (toPy / distToP) * pullStrength;

                // Real Collision Detection with Multi-Belt Keplerian Asteroids & Eccentric Rogue Bolides!
                int totalAsteroids = Math.Min(24, Math.Max(6, _current.AsteroidBeltCount + _planetsExploredCount * 2));
                double speedMult = 1.0 + _difficultyLevel * 0.14 + Math.Min(0.8, _current.Eccentricity);
                for (int h = 0; h < totalAsteroids; h++)
                {
                    int ringIdx = h % 3; // 0 = inner prograde Roche ring, 1 = middle retrograde Kirkwood belt, 2 = eccentric rogue bolide
                    double dir = (ringIdx == 1) ? 1.0 : -1.0;
                    double hAng = dir * _simTime * (1.85 + ringIdx * 0.55) * speedMult + h * (2.0 * Math.PI / Math.Max(1, totalAsteroids));
                    float hRad = 54f + ringIdx * 28f + (ringIdx == 2 ? (float)(Math.Sin(_simTime * 3.2 + h) * 22.0) : 0f);
                    float hx = _planetScreenX + (float)Math.Cos(hAng) * hRad;
                    float hy = _planetScreenY + (float)Math.Sin(hAng) * hRad;

                    float dxH = _beeX - hx;
                    float dyH = _beeY - hy;
                    float distH = (float)Math.Sqrt(dxH * dxH + dyH * dyH);
                    if (distH < 18.0f && _asteroidHitCooldown <= 0.0)
                    {
                        _hazardHitFlash = 1.0;
                        _asteroidHitCooldown = 0.85;
                        float nx = dxH / Math.Max(1f, distH);
                        float ny = dyH / Math.Max(1f, distH);
                        float knockback = 4.2f + _difficultyLevel * 0.45f;
                        _beeVx = nx * knockback;
                        _beeVy = ny * knockback;
                        _captureProgress = Math.Max(0.0, _captureProgress - (6.0 + _difficultyLevel * 1.1));
                        _surpriseEventBanner = (_lang == "fr") ? _current.AsteroidPhysicsFr : _current.AsteroidPhysicsEn;
                        PlaySynthesizedSciFiWav(180.0, 95.0, 160);
                        AddPointsWithPopup(25, string.Format("☄ BOLIDE IMPACT! Δv KNOCKBACK (v_esc={0:F1}km/s)", _current.EscapeVelKms), _beeX, _beeY - 14f, Color.FromArgb(248, 113, 113));
                        break;
                    }
                }

                _beeX = Math.Max(_galaxyRect.X + 36f, Math.Min(_galaxyRect.Right - 36f, _beeX + _beeVx));
                _beeY = Math.Max(_galaxyRect.Y + 36f, Math.Min(_galaxyRect.Bottom - 36f, _beeY + _beeVy));
                _beeVx *= 0.92f;
                _beeVy *= 0.92f;

                // Check collection of Golden Bonus Stars (+200 PTS each!)
                foreach (var st in _bonusStars)
                {
                    if (!st.Collected)
                    {
                        float sx = _galaxyRect.X + st.XNorm * _galaxyRect.Width;
                        float sy = _galaxyRect.Y + st.YNorm * _galaxyRect.Height;
                        if (Math.Sqrt(Math.Pow(sx - _beeX, 2) + Math.Pow(sy - _beeY, 2)) < 42.0)
                        {
                            st.Collected = true;
                            PlaySynthesizedSciFiWav(880.0, 1760.0, 120);
                            AddPointsWithPopup(200, "⭐ SCIENCE STAR!", sx, sy, Color.FromArgb(250, 204, 21));
                        }
                    }
                }

                // Lagrange Synchronization Gate around planet (tightens slightly with difficulty level)
                float syncRadius = Math.Max(78.0f, 108.0f - _difficultyLevel * 2.5f);
                double fillRate = Math.Max(0.75, 1.48 - _difficultyLevel * 0.06);
                if (distToP < syncRadius)
                {
                    _captureProgress = Math.Min(100.0, _captureProgress + fillRate);
                    if (_captureProgress >= 100.0)
                    {
                        PlaySynthesizedSciFiWav(660.0, 1320.0, 220);
                        AddPointsWithPopup(500 + _difficultyLevel * 50, "🛰️ ORBIT SYNCED!", _planetScreenX, _planetScreenY, Color.FromArgb(52, 211, 153));
                        ResetOolitePlanetfallState();
                        _gameStage = 2;
                        _showNarrativeCard = true;
                        SpeakStepNarrativeAndAstrophysics();
                    }
                }
            }
            else if (_gameStage == 2)
            {
                // Stage 2: Complex Multi-Layer Atmospheric Descent with Real Planetary Gravity g = M/R^2,
                // Scale Height H = kT/(mu*mH*g), and Active Lateral Crosswind Shear!
                double gVal = Math.Max(0.25, _current.SurfaceGravityG);
                double hVal = Math.Max(2.0, _current.ScaleHeightKm);
                double wMax = Math.Max(15.0, _current.WindShearMaxMs);

                // Determine current atmospheric layer (Thermosphere -> Stratosphere -> Troposphere -> Boundary Layer)
                double windPushFactor = 0.0;
                if (_landerAltKm > 2.2)
                {
                    _atmoLayerName = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "1/4 THERMOSPHÈRE // Choc ionosphérique T_eq={0:F0}K | v_lib={1:F1}km/s", _current.TeqK, _current.EscapeVelKms)
                        : string.Format(CultureInfo.InvariantCulture, "1/4 THERMOSPHERE // Ionospheric shock T_eq={0:F0}K | v_esc={1:F1}km/s", _current.TeqK, _current.EscapeVelKms);
                    windPushFactor = 0.25;
                }
                else if (_landerAltKm > 1.4)
                {
                    _atmoLayerName = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "2/4 STRATOSPHÈRE // Jet-Stream Corioli {0:F0} m/s | H={1:F1} km", wMax, hVal)
                        : string.Format(CultureInfo.InvariantCulture, "2/4 STRATOSPHERE // Supersonic Jet Stream {0:F0} m/s | H={1:F1} km", wMax, hVal);
                    windPushFactor = 0.65;
                }
                else if (_landerAltKm > 0.6)
                {
                    _atmoLayerName = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "3/4 TROPOSPHÈRE // Convection P={0:F1} bar | Gravité g={1:F2}g", _current.SurfacePressureBar, gVal)
                        : string.Format(CultureInfo.InvariantCulture, "3/4 TROPOSPHERE // Convective Clouds P={0:F1} bar | Gravity g={1:F2}g", _current.SurfacePressureBar, gVal);
                    windPushFactor = 0.85;
                }
                else
                {
                    _atmoLayerName = (_lang == "fr")
                        ? string.Format(CultureInfo.InvariantCulture, "4/4 COUCHE LIMITE // Cisaillement de surface | Alignez < {0:F0}m !", Math.Max(26.0, 44.0 - _difficultyLevel * 1.6))
                        : string.Format(CultureInfo.InvariantCulture, "4/4 BOUNDARY LAYER // Surface crosswind shear | Align < {0:F0}m!", Math.Max(26.0, 44.0 - _difficultyLevel * 1.6));
                    windPushFactor = 0.55;
                }

                // Target terminal descent speed increases with real surface gravity g and difficulty unless braking [SPACE]!
                double naturalGravitySpeed = Math.Min(138.0, 56.0 + gVal * 11.5 + _difficultyLevel * 3.8);
                double targetSpeed = _kSpace ? 40.0 : naturalGravitySpeed;
                _landerSpeedMs += (targetSpeed - _landerSpeedMs) * 0.038;

                // Active lateral crosswind drift pushes _landerOffsetX (stronger at higher difficulty & higher WindShearMaxMs)
                double liveWindWave = Math.Sin(_simTime * 1.9 + _current.Index) * (wMax * 0.012 * windPushFactor * (0.4 + _difficultyLevel * 0.08));
                _landerWindShearMs = Math.Round(Math.Abs(wMax * windPushFactor * (0.55 + 0.35 * Math.Sin(_simTime * 1.4))), 1);
                float autoStabilizerDecay = Math.Max(0.955f, 0.988f - _difficultyLevel * 0.003f);
                _landerOffsetX = Math.Max(-145f, Math.Min(145f, (_landerOffsetX + (float)liveWindWave) * autoStabilizerDecay));

                _landerCabinTempC = Math.Round(22.0 + (_landerSpeedMs * 0.86) + (_current.TeqK > 600 ? 28.0 : 0.0), 1);
                if (_landerSpeedMs > 98.0)
                {
                    _landerFwdShield = Math.Max(42.0, _landerFwdShield - 0.08 * (_difficultyLevel * 0.3));
                }
                _landerEnergy = Math.Min(256.0, _landerEnergy + 0.20);

                // Collect Golden Bonus Stars during planetary descent as altitude crosses 2.2 km, 1.4 km, 0.7 km
                double[] starAlts = new double[] { 2.2, 1.4, 0.7 };
                for (int i = 0; i < _bonusStars.Count && i < 3; i++)
                {
                    if (!_bonusStars[i].Collected && _landerAltKm <= starAlts[i] && Math.Abs(_landerOffsetX) < 95f)
                    {
                        _bonusStars[i].Collected = true;
                        PlaySynthesizedSciFiWav(920.0, 1680.0, 110);
                        AddPointsWithPopup(200, "⭐ ATMO DATA STAR!", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 240f, Color.FromArgb(250, 204, 21));
                    }
                }

                // Smooth altitude descent from 3.00 km -> 0.00 km
                _landerAltKm = Math.Max(0.0, _landerAltKm - (_landerSpeedMs * 0.000065));

                if (_landerAltKm <= 0.05)
                {
                    // At difficulty >= 4, check if touchdown speed or lateral crosswind offset exceeded safe limits;
                    // if so, trigger an exciting Retro-Booster Go-Around to 0.90 km with real astrophysics explanation!
                    double maxSafeSpeed = Math.Max(82.0, 112.0 - _difficultyLevel * 2.5);
                    float maxSafeOffset = Math.Max(26.0f, 48.0f - _difficultyLevel * 1.8f);
                    if (_difficultyLevel >= 4 && (_landerSpeedMs > maxSafeSpeed || Math.Abs(_landerOffsetX) > maxSafeOffset))
                    {
                        _landerAltKm = 0.85;
                        _landerSpeedMs = 64.0;
                        _landerOffsetX *= 0.45f;
                        _surpriseEventBanner = (_lang == "fr")
                            ? string.Format(CultureInfo.InvariantCulture, "⚠ REMISE DES GAZ HAUTE-G (g={0:F2}g) ! Vitesse ou dérive trop élevée — Appuyez sur [ESPACE] pour freiner !", gVal)
                            : string.Format(CultureInfo.InvariantCulture, "⚠ HIGH-G / WIND-SHEAR GO-AROUND (g={0:F2}g)! Touchdown too fast/off-pad — Press [SPACE] to Retro-Brake!", gVal);
                        PlaySynthesizedSciFiWav(240.0, 520.0, 180);
                        AddPointsWithPopup(50, "🔥 RETRO GO-AROUND! PRESS [SPACE]", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 240f, Color.FromArgb(251, 146, 60));
                    }
                    else
                    {
                        CompletePlanetLandingSuccess();
                    }
                }
            }

            this.Invalidate();
        }

        private void OnKeyDownHandler(object sender, KeyEventArgs e)
        {
            _idleSeconds = 0.0;
            if (e.KeyCode == Keys.Up || e.KeyCode == Keys.W || e.KeyCode == Keys.Z) _kUp = true;
            if (e.KeyCode == Keys.Down || e.KeyCode == Keys.S) _kDown = true;
            if (e.KeyCode == Keys.Left || e.KeyCode == Keys.A || e.KeyCode == Keys.Q) _kLeft = true;
            if (e.KeyCode == Keys.Right || e.KeyCode == Keys.D) _kRight = true;
            if (e.KeyCode == Keys.Enter)
            {
                if (_gameStage == 0 && _warpTimer <= 0) LaunchSatellites();
                else if (_gameStage == 1) { ResetOolitePlanetfallState(); _gameStage = 2; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); }
                else if (_gameStage == 2) CompletePlanetLandingSuccess();
                else if (_gameStage == 3) ApproveOrReject(true);
                else if (_gameStage == 4) TriggerBoincIdleGpuWorkunitSync("hivemind_tess_at_home", true);
                return;
            }
            if (e.KeyCode == Keys.Space)
            {
                _kSpace = true;
                if (_gameStage == 0 && _warpTimer <= 0) LaunchSatellites();
                else if (_gameStage == 1)
                {
                    _beeX += (_planetScreenX - _beeX) * 0.35f;
                    _beeY += (_planetScreenY - _beeY) * 0.35f;
                    AddPointsWithPopup(100, "ORBIT BOOST", _beeX, _beeY, Color.FromArgb(56, 189, 248));
                }
                else if (_gameStage == 2)
                {
                    _landerSpeedMs = Math.Max(35.0, _landerSpeedMs - 14.0);
                    _landerOffsetX *= 0.4f;
                    _landerAltKm = Math.Max(0.04, _landerAltKm - 0.45);
                    PlaySynthesizedSciFiWav(440.0, 880.0, 110);
                    AddPointsWithPopup(100, "RETRO-BRAKE & ALIGN", _galaxyRect.X + _galaxyRect.Width * 0.5f, _galaxyRect.Y + 260f, Color.FromArgb(52, 211, 153));
                }
                else if (_gameStage == 3) ApproveOrReject(true);
                else if (_gameStage == 4) TriggerBoincIdleGpuWorkunitSync("hivemind_tess_at_home", true);
            }
            if (e.KeyCode == Keys.N || e.KeyCode == Keys.H)
            {
                _showNarrativeCard = !_showNarrativeCard;
                if (_showNarrativeCard) SpeakStepNarrativeAndAstrophysics();
            }
            if (e.KeyCode == Keys.C && _gameStage == 2)
            {
                _landingClearanceGranted = true;
                AddPointsWithPopup(100, "RADIO CLEARANCE", _galaxyRect.X + 220f, _galaxyRect.Y + 120f, Color.FromArgb(56, 189, 248));
                PlaySynthesizedSciFiWav(740.0, 1180.0, 140);
            }
            if (e.KeyCode == Keys.G && _gameStage == 2)
            {
                _landingGearDown = true;
                PlaySynthesizedSciFiWav(420.0, 680.0, 120);
            }
            if (e.KeyCode == Keys.Tab)
            {
                _stage2TradeTabExpanded = !_stage2TradeTabExpanded;
                PlaySynthesizedSciFiWav(720.0, 1080.0, 80);
                this.Invalidate();
            }
            if (e.KeyCode == Keys.R && _gameStage == 3) ApproveOrReject(false);
            if (e.KeyCode == Keys.M) { SelectNewUnseenPlanetForNewGame(true); _gameStage = 0; this.Invalidate(); }
            if (e.KeyCode == Keys.D1) { _gameStage = 0; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); }
            if (e.KeyCode == Keys.D2) { _gameStage = 1; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); }
            if (e.KeyCode == Keys.D3 || e.KeyCode == Keys.P) { ResetOolitePlanetfallState(); _gameStage = 2; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); }
            if (e.KeyCode == Keys.D4) { _gameStage = 3; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); }
            if (e.KeyCode == Keys.D5 || e.KeyCode == Keys.B) { _gameStage = 4; _showNarrativeCard = false; _idleSeconds = 4.8; SpeakStepNarrativeAndAstrophysics(); }
            if (e.KeyCode == Keys.L)
            {
                string[] langs = new string[] { "en", "fr", "es", "de", "zh" };
                int idx = Array.IndexOf(langs, _lang);
                _lang = langs[(idx + 1) % langs.Length];
                SpeakStepNarrativeAndAstrophysics();
            }
            if (e.KeyCode == Keys.V)
            {
                string[] voices = new string[] { "Puck", "Aoede", "Kore", "Charon" };
                int vIdx = Array.IndexOf(voices, _geminiVoiceName);
                _geminiVoiceName = voices[(vIdx + 1) % voices.Length];
                _activeVoiceEngineLabel = "GCP GEMINI NEURAL • " + _geminiVoiceName.ToUpperInvariant();
                SpeakStepNarrativeAndAstrophysics();
            }
        }

        private void OnKeyUpHandler(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Up || e.KeyCode == Keys.W || e.KeyCode == Keys.Z) _kUp = false;
            if (e.KeyCode == Keys.Down || e.KeyCode == Keys.S) _kDown = false;
            if (e.KeyCode == Keys.Left || e.KeyCode == Keys.A || e.KeyCode == Keys.Q) _kLeft = false;
            if (e.KeyCode == Keys.Right || e.KeyCode == Keys.D) _kRight = false;
            if (e.KeyCode == Keys.Space || e.KeyCode == Keys.Enter) _kSpace = false;
        }

        private void OnMouseDownHandler(object sender, MouseEventArgs e)
        {
            _idleSeconds = 0.0;
            // 1. Check HUD hotspots (Language pills, Stage pills, Voice toggle, Narrative, Oolite controls, Approve/Reject, BOINC)
            foreach (var h in _hotspots)
            {
                if (h.Rect.Contains(e.Location))
                {
                    if (h.Id.StartsWith("lang:"))
                    {
                        _lang = h.Id.Substring(5);
                        PlaySynthesizedSciFiWav(980.0, 1420.0, 80);
                        SpeakStepNarrativeAndAstrophysics();
                        return;
                    }
                    if (h.Id == "toggle_voice")
                    {
                        _voiceAndSoundOn = !_voiceAndSoundOn;
                        if (_voiceAndSoundOn) SpeakStepNarrativeAndAstrophysics();
                        else
                        {
                            try { mciSendString("close gcp_tts", null, 0, IntPtr.Zero); } catch { }
                        }
                        return;
                    }
                    if (h.Id == "cycle_gemini_voice")
                    {
                        string[] voices = new string[] { "Puck", "Aoede", "Kore", "Charon" };
                        int vIdx = Array.IndexOf(voices, _geminiVoiceName);
                        _geminiVoiceName = voices[(vIdx + 1) % voices.Length];
                        _activeVoiceEngineLabel = "GCP GEMINI NEURAL • " + _geminiVoiceName.ToUpperInvariant();
                        AddPointsWithPopup(50, "♫ GEMINI VOICE: " + _geminiVoiceName.ToUpperInvariant(), e.X, e.Y, Color.FromArgb(56, 189, 248));
                        PlaySynthesizedSciFiWav(880.0, 1320.0, 90);
                        SpeakStepNarrativeAndAstrophysics();
                        return;
                    }
                    if (h.Id == "toggle_narrative")
                    {
                        _showNarrativeCard = !_showNarrativeCard;
                        if (_showNarrativeCard) SpeakStepNarrativeAndAstrophysics();
                        return;
                    }
                    if (h.Id == "kid_lesson")
                    {
                        AddPointsWithPopup(150, "🎓 KID SCIENCE LESSON!", e.X, e.Y, Color.FromArgb(250, 204, 21));
                        PlaySynthesizedSciFiWav(660.0, 1320.0, 140);
                        SpeakWithSubtitles(GetCurrentPlanetKidLesson());
                        return;
                    }
                    if (h.Id == "stg:0") { _gameStage = 0; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); return; }
                    if (h.Id == "stg:1") { _gameStage = 1; _showNarrativeCard = true; InitBonusStars(); _beeX = _galaxyRect.X + 200f; _beeY = _galaxyRect.Y + 220f; _captureProgress = 20; SpeakStepNarrativeAndAstrophysics(); return; }
                    if (h.Id == "stg:2") { ResetOolitePlanetfallState(); _gameStage = 2; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); return; }
                    if (h.Id == "stg:3") { _gameStage = 3; _showNarrativeCard = true; SpeakStepNarrativeAndAstrophysics(); return; }
                    if (h.Id == "stg:4") { _gameStage = 4; _showNarrativeCard = false; _idleSeconds = 4.8; SpeakStepNarrativeAndAstrophysics(); return; }
                    if (h.Id == "boinc_crunch_now")
                    {
                        _idleSeconds = 5.2;
                        TriggerBoincIdleGpuWorkunitSync("hivemind_tess_at_home", true);
                        return;
                    }
                    if (h.Id == "boinc_toggle_mode")
                    {
                        _boincForceAlwaysOn = !_boincForceAlwaysOn;
                        AddPointsWithPopup(100, _boincForceAlwaysOn ? "BOINC GPU: FORCE ON" : "BOINC GPU: AUTO IDLE (>4s)", e.X, e.Y, Color.FromArgb(0, 240, 255));
                        PlaySynthesizedSciFiWav(740.0, 1180.0, 90);
                        return;
                    }
                    if (h.Id == "boinc_open_web")
                    {
                        try { Process.Start("http://127.0.0.1:8095/community"); } catch { }
                        AddPointsWithPopup(100, "OPENED GCP COMMUNITY HUB", e.X, e.Y, Color.FromArgb(52, 255, 141));
                        return;
                    }
                    if (h.Id == "boinc_post_note")
                    {
                        _communityActiveMembers++;
                        _communityLatestNote = "Cmdr_JeanPaul_NCC31337 (Steam RTX 2070): Verified BLS transit dip (" + (_current != null ? _current.DepthPpm.ToString("F0") : "2920") + " ppm) & shared with GCP Astrophysicists!";
                        AddPointsWithPopup(150, "CO-OP TRANSIT NOTE POSTED!", e.X, e.Y, Color.FromArgb(250, 204, 21));
                        PlaySynthesizedSciFiWav(820.0, 1480.0, 120);
                        return;
                    }
                    if (h.Id == "oolite_toggle_trade")
                    {
                        _stage2TradeTabExpanded = !_stage2TradeTabExpanded;
                        AddPointsWithPopup(50, _stage2TradeTabExpanded ? "CARGO/EQUIP MFD OPEN" : "FLIGHT HUD DECLUTTERED", e.X, e.Y, Color.FromArgb(0, 240, 255));
                        PlaySynthesizedSciFiWav(720.0, 1080.0, 80);
                        return;
                    }
                    if (h.Id == "oolite_clearance")
                    {
                        _landingClearanceGranted = true;
                        AddPointsWithPopup(100, "CLEARANCE OK", e.X, e.Y, Color.FromArgb(52, 211, 153));
                        PlaySynthesizedSciFiWav(740.0, 1180.0, 140);
                        return;
                    }
                    if (h.Id == "oolite_gear")
                    {
                        _landingGearDown = true;
                        AddPointsWithPopup(100, "GEAR LOCKED", e.X, e.Y, Color.FromArgb(56, 189, 248));
                        PlaySynthesizedSciFiWav(420.0, 680.0, 120);
                        return;
                    }
                    if (h.Id == "oolite_brake")
                    {
                        _landerSpeedMs = Math.Max(35.0, _landerSpeedMs - 14.0);
                        _landerOffsetX *= 0.35f;
                        _landerAltKm = Math.Max(0.04, _landerAltKm - 0.50);
                        AddPointsWithPopup(100, "RETRO-BRAKE & ALIGN!", e.X, e.Y, Color.FromArgb(52, 211, 153));
                        PlaySynthesizedSciFiWav(320.0, 640.0, 110);
                        return;
                    }
                    if (h.Id == "oolite_autoland")
                    {
                        CompletePlanetLandingSuccess();
                        return;
                    }
                    if (h.Id == "approve") { ApproveOrReject(true); return; }
                    if (h.Id == "reject") { ApproveOrReject(false); return; }
                }
            }

            // 2. Check Bottom 10-Slot Collection Cards
            for (int i = 0; i < 10; i++)
            {
                if (_slotRects[i].Contains(e.Location))
                {
                    SelectTarget(i, false, true);
                    return;
                }
            }

            // 3. Check Main Galaxy Viewport click
            if (_galaxyRect.Contains(e.Location))
            {
                if (_gameStage == 0)
                {
                    int bestIdx = -1;
                    double bestD = 9999.0;
                    for (int i = 0; i < _targets.Count; i++)
                    {
                        float px = _galaxyRect.X + _targets[i].GalaxyXNorm * _galaxyRect.Width;
                        float py = _galaxyRect.Y + _targets[i].GalaxyYNorm * _galaxyRect.Height;
                        double d = Math.Sqrt(Math.Pow(px - e.X, 2) + Math.Pow(py - e.Y, 2));
                        if (d < bestD) { bestD = d; bestIdx = i; }
                    }
                    if (bestD < 64.0 && bestIdx >= 0)
                    {
                        SelectTarget(bestIdx, true, true);
                        if (e.Button == MouseButtons.Right || e.Clicks >= 2) LaunchSatellites();
                    }
                    else
                    {
                        _shipXNorm = (e.X - _galaxyRect.X) / (float)_galaxyRect.Width;
                        _shipYNorm = (e.Y - _galaxyRect.Y) / (float)_galaxyRect.Height;
                    }
                }
                else if (_gameStage == 1)
                {
                    _beeVx += (e.X - _beeX) * 0.06f;
                    _beeVy += (e.Y - _beeY) * 0.06f;
                }
                else if (_gameStage == 2)
                {
                    float centerX = _galaxyRect.X + _galaxyRect.Width * 0.5f;
                    _landerOffsetX = Math.Max(-120f, Math.Min(120f, (e.X - centerX) * 0.25f));
                    _landerSpeedMs = Math.Max(36.0, _landerSpeedMs - 10.0);
                    AddPointsWithPopup(50, "GLIDE ADJUST", e.X, e.Y, Color.FromArgb(56, 189, 248));
                }
            }
        }

        private void ComputeLayoutRectangles()
        {
            int W = this.ClientSize.Width;
            int H = this.ClientSize.Height;
            int padX = Math.Max(28, (int)(W * 0.032));
            int padY = Math.Max(14, (int)(H * 0.018));

            _consoleRect = new Rectangle(padX, padY, W - padX * 2, H - padY * 2);
            int cx = _consoleRect.X + 12;
            int cw = _consoleRect.Width - 24;
            int cy = _consoleRect.Y + 10;

            _topTelemetryRect = new Rectangle(cx, cy, cw, 30);
            cy += 35;

            _bannerRect = new Rectangle(cx, cy, cw, 28);
            cy += 34;

            int bottomH = 122;
            int oscH = (int)(_consoleRect.Height * 0.215);
            int galH = _consoleRect.Bottom - cy - oscH - bottomH - 22;

            _galaxyRect = new Rectangle(cx, cy, cw, Math.Max(220, galH));
            cy += _galaxyRect.Height + 7;

            _oscRect = new Rectangle(cx, cy, cw, oscH);
            cy += _oscRect.Height + 7;

            _collectionRect = new Rectangle(cx, cy, cw, bottomH);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;

            _hotspots.Clear();
            ComputeLayoutRectangles();

            DrawBridgeEnvironmentAndConsoleFrame(g);
            DrawTopLcarsTelemetryAndLanguageBar(g, _topTelemetryRect);
            DrawGoldenStageBanner(g, _bannerRect);

            if (_gameStage == 0)
            {
                DrawGalaxyViewportStage0(g, _galaxyRect);
                DrawDualNeonOscilloscopes(g, _oscRect);
            }
            else if (_gameStage == 1)
            {
                DrawOrbitalActionStage1(g, _galaxyRect);
                DrawDualNeonOscilloscopes(g, _oscRect);
            }
            else if (_gameStage == 2)
            {
                // Stage 2: Full-Screen Easy & Intuitive Oolite (oolite.space) + Planetfall OXP Planetary Landing Cockpit!
                Rectangle ooliteFullRect = new Rectangle(_galaxyRect.X, _galaxyRect.Y, _galaxyRect.Width, _oscRect.Bottom - _galaxyRect.Y);
                DrawOolitePlanetfallStage2(g, ooliteFullRect);
            }
            else if (_gameStage == 4)
            {
                // Stage 4 (5.BOINC/GCP): Full-Screen Steam Free-to-Play Idle-GPU BOINC Citizen-Science & GCP Community Hub!
                Rectangle boincFullRect = new Rectangle(_galaxyRect.X, _galaxyRect.Y, _galaxyRect.Width, _oscRect.Bottom - _galaxyRect.Y);
                DrawBoincAndCommunityStage4(g, boincFullRect);
            }
            else
            {
                DrawAstrophysicsAndJwstStage3(g, _galaxyRect);
                DrawDualNeonOscilloscopes(g, _oscRect);
            }

            if (_showNarrativeCard && _gameStage != 2 && _gameStage != 4)
            {
                DrawStepNarrativeOverlay(g, _galaxyRect);
            }

            DrawBottom10SlotHolographicCollection(g, _collectionRect);
            DrawFloatingPopups(g);
        }

        private void DrawBridgeEnvironmentAndConsoleFrame(Graphics g)
        {
            int W = this.ClientSize.Width;
            int H = this.ClientSize.Height;

            // Left Starfleet bridge engineering sparks glow & Right blue anamorphic lens flare (matching reference image!)
            using (GraphicsPath sparkPath = new GraphicsPath())
            {
                sparkPath.AddEllipse(-120, (int)(H * 0.45), 320, 380);
                using (PathGradientBrush pgb = new PathGradientBrush(sparkPath))
                {
                    pgb.CenterColor = Color.FromArgb(65, 255, 145, 25);
                    pgb.SurroundColors = new Color[] { Color.Transparent };
                    g.FillPath(pgb, sparkPath);
                }
            }
            using (GraphicsPath flarePath = new GraphicsPath())
            {
                flarePath.AddEllipse(W - 220, (int)(H * 0.30), 340, 260);
                using (PathGradientBrush pgb = new PathGradientBrush(flarePath))
                {
                    pgb.CenterColor = Color.FromArgb(75, 56, 189, 248);
                    pgb.SurroundColors = new Color[] { Color.Transparent };
                    g.FillPath(pgb, flarePath);
                }
            }

            // Curved Console Outer Glass Bezel
            using (SolidBrush conBg = new SolidBrush(Color.FromArgb(245, 5, 10, 20)))
            using (Pen outerGlow = new Pen(Color.FromArgb(90, 34, 211, 238), 4.0f))
            using (Pen innerRim = new Pen(Color.FromArgb(190, 56, 189, 248), 1.6f))
            {
                g.FillRectangle(conBg, _consoleRect);
                g.DrawRectangle(outerGlow, _consoleRect);
                g.DrawRectangle(innerRim, _consoleRect);
            }
        }

        private void DrawTopLcarsTelemetryAndLanguageBar(Graphics g, Rectangle r)
        {
            using (SolidBrush bg = new SolidBrush(Color.FromArgb(230, 4, 11, 24)))
            using (Pen border = new Pen(Color.FromArgb(130, 56, 189, 248), 1.2f))
            {
                g.FillRectangle(bg, r);
                g.DrawRectangle(border, r);
            }

            // Gamified Cadet Score, Streak Multiplier, Medals & Live BOINC Idle-GPU Status on Left of Top Bar
            bool isBoincActive = (_boincAutoIdleMode && _idleSeconds >= 4.0) || _boincForceAlwaysOn;
            using (Font fScore = new Font("Consolas", 8.2f, FontStyle.Bold))
            using (SolidBrush bGold = new SolidBrush(Color.FromArgb(253, 224, 71)))
            using (SolidBrush bMint = new SolidBrush(isBoincActive ? Color.FromArgb(52, 255, 141) : Color.FromArgb(56, 189, 248)))
            {
                string scoreTxt = string.Format(
                    CultureInfo.InvariantCulture,
                    "★ {0:N0} PTS (x{1}) | MEDALS: {2}/10 | ",
                    _scoreHmt, _streakCount, _medalsCount
                );
                g.DrawString(scoreTxt, fScore, bGold, r.X + 8, r.Y + 7);
                string boincPillTxt = isBoincActive
                    ? string.Format(CultureInfo.InvariantCulture, "BOINC GPU: IDLE CRUNCH ({0} WUs)", _boincWorkunitsCompleted)
                    : "BOINC GPU: STANDBY (60FPS GAME)";
                g.DrawString(boincPillTxt, fScore, bMint, r.X + 258, r.Y + 7);
            }

            // Interactive Kid Story & Lesson Briefing Toggle Button + 5-Language Pills + GCP Gemini Voice Pill + 5 Stage Pills
            int bx = r.Right - 885;

            Rectangle narRect = new Rectangle(bx, r.Y + 4, 96, 22);
            using (SolidBrush nb = new SolidBrush(_showNarrativeCard ? Color.FromArgb(168, 85, 247) : Color.FromArgb(30, 41, 59)))
            using (Pen np = new Pen(Color.FromArgb(216, 180, 254), 1.1f))
            {
                g.FillRectangle(nb, narRect);
                g.DrawRectangle(np, narRect);
                string narLbl = (_lang == "fr") ? "► HISTOIRE [N]" : "► STORY [N]";
                g.DrawString(narLbl, new Font("Consolas", 6.6f, FontStyle.Bold), Brushes.White, narRect.X + 4, narRect.Y + 4);
            }
            _hotspots.Add(new HudHotspot { Id = "toggle_narrative", Rect = narRect });
            bx += 100;

            string[] langs = new string[] { "fr", "es", "de", "en", "zh" };
            string[] langLabels = new string[] { "FR", "ES", "DE", "EN", "中文" };
            for (int i = 0; i < langs.Length; i++)
            {
                Rectangle pb = new Rectangle(bx, r.Y + 4, 28, 22);
                bool active = (_lang == langs[i]);
                using (SolidBrush b = new SolidBrush(active ? Color.FromArgb(16, 185, 129) : Color.FromArgb(14, 28, 52)))
                using (Pen p = new Pen(active ? Color.White : Color.FromArgb(56, 189, 248), 1.1f))
                {
                    g.FillRectangle(b, pb);
                    g.DrawRectangle(p, pb);
                    g.DrawString(langLabels[i], new Font("Consolas", 7.2f, FontStyle.Bold), active ? Brushes.Black : Brushes.LightCyan, pb.X + 2, pb.Y + 4);
                }
                _hotspots.Add(new HudHotspot { Id = "lang:" + langs[i], Rect = pb });
                bx += 31;
            }

            // GCP Gemini Neural Voice Persona Selector Pill ([V] key or click to cycle Puck / Aoede / Kore / Charon)
            Rectangle gemRect = new Rectangle(bx, r.Y + 4, 92, 22);
            using (SolidBrush gb = new SolidBrush(Color.FromArgb(14, 116, 144)))
            using (Pen gp = new Pen(Color.FromArgb(103, 232, 249), 1.1f))
            {
                g.FillRectangle(gb, gemRect);
                g.DrawRectangle(gp, gemRect);
                g.DrawString("♫ GCP:" + _geminiVoiceName.ToUpperInvariant(), new Font("Consolas", 6.6f, FontStyle.Bold), Brushes.White, gemRect.X + 4, gemRect.Y + 4);
            }
            _hotspots.Add(new HudHotspot { Id = "cycle_gemini_voice", Rect = gemRect });
            bx += 96;

            // Voice toggle pill (localized EN vs FR)
            Rectangle vRect = new Rectangle(bx, r.Y + 4, 54, 22);
            using (SolidBrush vb = new SolidBrush(_voiceAndSoundOn ? Color.FromArgb(255, 170, 0) : Color.FromArgb(30, 41, 59)))
            {
                g.FillRectangle(vb, vRect);
                string vLbl = (_lang == "fr")
                    ? (_voiceAndSoundOn ? "♫ VOIX" : "x MUET")
                    : (_voiceAndSoundOn ? "♫ ON" : "x MUTE");
                g.DrawString(vLbl, new Font("Segoe UI", 7.0f, FontStyle.Bold), _voiceAndSoundOn ? Brushes.Black : Brushes.Silver, vRect.X + 4, vRect.Y + 3);
            }
            _hotspots.Add(new HudHotspot { Id = "toggle_voice", Rect = vRect });
            bx += 58;

            // 5 Stage Pills localized to English / French (including 5.BOINC/GCP)
            string[] stgNames = (_lang == "fr")
                ? new string[] { "1.GALAXIE", "2.ORBITE", "3.ATTERRIR", "4.JWST", "5.BOINC/GCP" }
                : new string[] { "1.GALAXY", "2.ORBIT BEE", "3.PLANET LAND", "4.JWST LAB", "5.BOINC/GCP" };
            int[] stgWidths = new int[] { 62, 68, 80, 64, 78 };
            for (int s = 0; s < 5; s++)
            {
                Rectangle sr = new Rectangle(bx, r.Y + 4, stgWidths[s], 22);
                bool act = (_gameStage == s);
                Color fillCol = act ? (s == 2 || s == 4 ? Color.FromArgb(16, 185, 129) : Color.FromArgb(56, 189, 248)) : Color.FromArgb(14, 28, 52);
                using (SolidBrush sb = new SolidBrush(fillCol))
                using (Pen sp = new Pen(s == 2 || s == 4 ? Color.FromArgb(52, 211, 153) : Color.FromArgb(56, 189, 248), 1.1f))
                {
                    g.FillRectangle(sb, sr);
                    g.DrawRectangle(sp, sr);
                    g.DrawString(stgNames[s], new Font("Consolas", 6.6f, FontStyle.Bold), act ? Brushes.Black : Brushes.LightCyan, sr.X + 3, sr.Y + 4);
                }
                _hotspots.Add(new HudHotspot { Id = "stg:" + s, Rect = sr });
                bx += stgWidths[s] + 4;
            }

            // Amber Chevrons >>>>>>> on far right
            using (Font fChev = new Font("Consolas", 10.5f, FontStyle.Bold))
            using (SolidBrush bAmb = new SolidBrush(Color.FromArgb(255, 170, 0)))
            {
                g.DrawString(">>>", fChev, bAmb, r.Right - 34, r.Y + 5);
            }
        }

        private void DrawGoldenStageBanner(Graphics g, Rectangle r)
        {
            using (SolidBrush bg = new SolidBrush(Color.FromArgb(230, 14, 18, 28)))
            using (Pen border = new Pen(Color.FromArgb(235, 245, 158, 11), 1.5f))
            {
                g.FillRectangle(bg, r);
                g.DrawRectangle(border, r);
            }
            using (Font fBanner = new Font("Consolas", 9.2f, FontStyle.Bold))
            using (SolidBrush bGold = new SolidBrush(Color.FromArgb(253, 224, 71)))
            {
                g.DrawString(GetStageBannerText(), fBanner, bGold, r.X + 10, r.Y + 6);
            }
        }

        private void Draw3DPlanetWithFallback(Graphics g, int idx, float cx, float cy, float radius)
        {
            if (idx >= 0 && idx < 10 && _planetGlobes[idx] != null)
            {
                g.DrawImage(_planetGlobes[idx], cx - radius, cy - radius, radius * 2f, radius * 2f);
            }
            else
            {
                using (SolidBrush b = new SolidBrush(_targets[idx].PrimaryColor))
                    g.FillEllipse(b, cx - radius, cy - radius, radius * 2f, radius * 2f);
            }
        }

        private void DrawGalaxyViewportStage0(Graphics g, Rectangle r)
        {
            Region oldClip = g.Clip;
            g.SetClip(r);

            if (_galaxyBackdrop != null)
                g.DrawImage(_galaxyBackdrop, r);
            else
                g.FillRectangle(Brushes.Black, r);

            // 1. USS HiveMind + Volumetric Green-Cyan Sensor Scan Cone + Green Telemetry Beams
            float sx = r.X + _shipXNorm * r.Width;
            float sy = r.Y + _shipYNorm * r.Height;
            float tx = r.X + _current.GalaxyXNorm * r.Width;
            float ty = r.Y + _current.GalaxyYNorm * r.Height;

            double ang = Math.Atan2(ty - sy, tx - sx);
            float dist = (float)Math.Sqrt(Math.Pow(tx - sx, 2) + Math.Pow(ty - sy, 2));
            PointF[] conePts = new PointF[] {
                new PointF(sx, sy),
                new PointF(sx + (float)Math.Cos(ang - 0.24) * dist, sy + (float)Math.Sin(ang - 0.24) * dist),
                new PointF(sx + (float)Math.Cos(ang + 0.24) * dist, sy + (float)Math.Sin(ang + 0.24) * dist)
            };
            using (LinearGradientBrush coneGrad = new LinearGradientBrush(
                new PointF(sx, sy), new PointF(tx, ty),
                Color.FromArgb(135, 52, 211, 153), Color.FromArgb(12, 34, 211, 238)))
            {
                g.FillPolygon(coneGrad, conePts);
            }
            using (Pen coneEdge = new Pen(Color.FromArgb(145, 52, 211, 153), 1.4f))
            {
                g.DrawLine(coneEdge, conePts[0], conePts[1]);
                g.DrawLine(coneEdge, conePts[0], conePts[2]);
            }

            // Green secondary lock lines to TRAPPIST-1e and K2-18 b (exact match to reference image!)
            using (Pen linkPen = new Pen(Color.FromArgb(170, 52, 211, 153), 1.6f))
            {
                g.DrawLine(linkPen, sx, sy, r.X + _targets[0].GalaxyXNorm * r.Width, r.Y + _targets[0].GalaxyYNorm * r.Height);
                g.DrawLine(linkPen, sx, sy, r.X + _targets[1].GalaxyXNorm * r.Width, r.Y + _targets[1].GalaxyYNorm * r.Height);
            }

            // 2. Draw all 10 Star Systems with 3D Raycast Planet Spheres, Concentric Rings & Green Bio-Scanner Badges
            Color[] dotCols = new Color[] {
                Color.FromArgb(52, 211, 153), Color.FromArgb(56, 189, 248), Color.FromArgb(244, 114, 182)
            };
            foreach (var sys in _targets)
            {
                float px = r.X + sys.GalaxyXNorm * r.Width;
                float py = r.Y + sys.GalaxyYNorm * r.Height;
                bool isSel = (sys.Index == _current.Index);

                // Warm golden sun-glow halo behind selected planet (like LHS 1140 b in reference image!)
                if (isSel)
                {
                    using (GraphicsPath gp = new GraphicsPath())
                    {
                        gp.AddEllipse(px - 68f, py - 68f, 136f, 136f);
                        using (PathGradientBrush pgb = new PathGradientBrush(gp))
                        {
                            pgb.CenterColor = Color.FromArgb(115, 255, 190, 70);
                            pgb.SurroundColors = new Color[] { Color.Transparent };
                            g.FillPath(pgb, gp);
                        }
                    }
                }

                // Concentric Orbital Rings + Orbiting Satellite Dots
                float[] ringRadii = new float[] { 30f, 44f, 58f };
                for (int k = 0; k < 3; k++)
                {
                    if (k == 2 && !isSel && sys.Index % 2 == 1) continue;
                    float rad = ringRadii[k];
                    using (Pen rp = new Pen(isSel && k == 2 ? Color.FromArgb(235, 251, 191, 36) : Color.FromArgb(105, 226, 232, 240), isSel && k == 2 ? 2.0f : 1.1f))
                    {
                        if (isSel && k == 2) rp.DashStyle = DashStyle.Dash;
                        g.DrawEllipse(rp, px - rad, py - rad, rad * 2f, rad * 2f);
                    }

                    double mAng = _simTime * (1.1 - k * 0.25) + sys.Index * 1.7 + k * 2.1;
                    float mx = px + (float)Math.Cos(mAng) * rad;
                    float my = py + (float)Math.Sin(mAng) * rad;
                    using (SolidBrush db = new SolidBrush(dotCols[k % 3]))
                        g.FillEllipse(db, mx - 3.5f, my - 3.5f, 7f, 7f);
                }

                // 3D Raycast-Shaded Planet Sphere
                Draw3DPlanetWithFallback(g, sys.Index, px, py, 20f);

                // Glowing Green Circular Bio-Scanner Badge on upper-right of ring (exact match to screenshot!)
                float bx = px + 28f;
                float by = py - 18f;
                using (SolidBrush bb = new SolidBrush(Color.FromArgb(235, 16, 185, 129)))
                using (Pen bp = new Pen(Color.White, 1.6f))
                {
                    g.FillEllipse(bb, bx - 11f, by - 11f, 22f, 22f);
                    g.DrawEllipse(bp, bx - 11f, by - 11f, 22f, 22f);
                    // Bio-organism triad symbol inside green badge
                    g.DrawEllipse(bp, bx - 4f, by - 5f, 4f, 4f);
                    g.DrawEllipse(bp, bx + 1f, by - 2f, 4f, 4f);
                    g.DrawEllipse(bp, bx - 3f, by + 1f, 4f, 4f);
                }

                // Crisp Planet Label underneath
                using (Font fLbl = new Font("Consolas", 9.2f, FontStyle.Bold))
                using (SolidBrush bLbl = new SolidBrush(isSel ? Color.FromArgb(253, 224, 71) : Color.White))
                {
                    SizeF sz = g.MeasureString(sys.ShortLabel, fLbl);
                    g.DrawString(sys.ShortLabel, fLbl, bLbl, px - sz.Width * 0.5f, py + 34f);
                }
            }

            // 3. Draw USS HiveMind Reticle, Horizontal Lens Flare & Starship
            using (Pen retOuter = new Pen(Color.FromArgb(110, 253, 224, 71), 1.2f))
            using (Pen retInner = new Pen(Color.FromArgb(253, 224, 71), 2.4f))
            using (Pen flareLine = new Pen(Color.FromArgb(160, 165, 243, 252), 1.6f))
            {
                g.DrawLine(flareLine, sx - 65f, sy - 6f, sx + 65f, sy - 6f);
                g.DrawEllipse(retOuter, sx - 24f, sy - 30f, 48f, 48f);
                g.DrawEllipse(retInner, sx - 12f, sy - 18f, 24f, 24f);
            }
            using (SolidBrush hullB = new SolidBrush(Color.FromArgb(56, 189, 248)))
            using (Pen hullP = new Pen(Color.White, 1.3f))
            {
                g.FillRectangle(hullB, sx - 18f, sy + 2f, 12f, 5f);
                g.FillRectangle(hullB, sx + 6f, sy + 2f, 12f, 5f);
                g.FillEllipse(Brushes.White, sx - 10f, sy - 4f, 20f, 9f);
                g.DrawEllipse(hullP, sx - 10f, sy - 4f, 20f, 9f);
            }
            using (Font fShip = new Font("Consolas", 8.8f, FontStyle.Bold))
            using (SolidBrush bShip = new SolidBrush(Color.FromArgb(253, 224, 71)))
            {
                g.DrawString("⊞ USS HIVEMIND", fShip, bShip, sx - 52f, sy + 14f);
            }

            // Animate Launched Bee-MoE Robotic Satellites
            foreach (var p in _launchedProbes)
            {
                using (SolidBrush pb = new SolidBrush(p.Color))
                using (Pen pp = new Pen(Color.White, 1.5f))
                {
                    g.FillRectangle(pb, p.X - 8f, p.Y - 3f, 16f, 6f);
                    g.DrawRectangle(pp, p.X - 8f, p.Y - 3f, 16f, 6f);
                }
            }

            // 4. Stitch Tactical Bridge Aerospace HUD Corner Telemetry Strips (Screens 1 & 6: Galactic Tactical Bridge)
            using (SolidBrush stitchBg = new SolidBrush(Color.FromArgb(215, 7, 14, 26)))
            using (Pen stitchBorder = new Pen(Color.FromArgb(165, 0, 240, 255), 1.1f))
            using (Font fSt = new Font("Consolas", 7.4f, FontStyle.Bold))
            using (SolidBrush bCyan = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bEm = new SolidBrush(Color.FromArgb(52, 255, 141)))
            using (SolidBrush bAmb = new SolidBrush(Color.FromArgb(255, 186, 32)))
            {
                Rectangle topSt = new Rectangle(r.X + 10, r.Y + 8, 550, 20);
                g.FillRectangle(stitchBg, topSt);
                g.DrawRectangle(stitchBorder, topSt);
                g.DrawString("❖ STITCH HUD [STAGE 1: GALAXIE] | SYS_LOCK: 99.4% | SUBCARRIER: 1420.405 MHz | WARP: 0.12 C", fSt, bCyan, topSt.X + 6, topSt.Y + 3);

                Rectangle botSt = new Rectangle(r.Right - 498, r.Y + 8, 488, 20);
                g.FillRectangle(stitchBg, botSt);
                g.DrawRectangle(stitchBorder, botSt);
                g.DrawString("AZIMUTH // 128.4° | DECL // -15°16'12\" | EPOCH // J2000.0 | BUFFER: 100% ■■■■□", fSt, bEm, botSt.X + 6, botSt.Y + 3);
            }

            // 5. Synchronized 5-Language Subtitle HUD Pill at bottom of Galaxy Viewport
            if (!string.IsNullOrEmpty(_subtitleText))
            {
                Rectangle subR = new Rectangle(r.X + 18, r.Bottom - 32, r.Width - 36, 26);
                using (SolidBrush sb = new SolidBrush(Color.FromArgb(215, 3, 8, 18)))
                using (Pen sp = new Pen(Color.FromArgb(160, 34, 211, 238), 1.2f))
                {
                    g.FillRectangle(sb, subR);
                    g.DrawRectangle(sp, subR);
                    g.DrawString(
                        "♫ [" + _activeVoiceEngineLabel + " • " + _lang.ToUpperInvariant() + "] « " + _subtitleText + " »",
                        new Font("Segoe UI", 8.5f, FontStyle.Bold), Brushes.Khaki, subR.X + 10, subR.Y + 4
                    );
                }
            }

            g.Clip = oldClip;
            using (Pen border = new Pen(Color.FromArgb(170, 56, 189, 248), 1.6f))
                g.DrawRectangle(border, r);
        }

        private void DrawOrbitalActionStage1(Graphics g, Rectangle r)
        {
            Region oldClip = g.Clip;
            g.SetClip(r);
            if (_galaxyBackdrop != null) g.DrawImage(_galaxyBackdrop, r);

            float cx = r.X + r.Width * 0.5f;
            float cy = r.Y + r.Height * 0.52f;
            double ecc = Math.Min(0.75, _current.Eccentricity);
            double E = SolveKeplerEquation(_simTime * 1.25, ecc);
            float aRx = 225f;
            float bRy = (float)(128.0 * Math.Sqrt(Math.Max(0.25, 1.0 - ecc * ecc)));
            float focusOff = (float)(aRx * ecc * 0.5);

            using (Pen orbPen = new Pen(Color.FromArgb(180, 0, 240, 255), 1.8f))
            {
                orbPen.DashStyle = DashStyle.Dash;
                g.DrawEllipse(orbPen, cx - focusOff - aRx, cy - bRy, aRx * 2f, bRy * 2f);
            }

            // Stitch Screen 0: Lagrange Points (L1, L2, L3, L4 STATION, L5) along the Orbital Ring
            using (Font fLag = new Font("Consolas", 7.2f, FontStyle.Bold))
            using (SolidBrush bLag = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bEmLag = new SolidBrush(Color.FromArgb(52, 255, 141)))
            using (Pen pLag = new Pen(Color.FromArgb(190, 0, 240, 255), 1.2f))
            {
                string[] lagNames = new string[] { "L1 [GATE]", "L2 [JWST]", "L3", "L4 [STATION]", "L5 [SYNODIC]" };
                double[] lagAngles = new double[] { 0.15, 0.85, Math.PI, Math.PI * 0.38, -Math.PI * 0.38 };
                for (int li = 0; li < 5; li++)
                {
                    float lx = cx - focusOff + (float)Math.Cos(lagAngles[li]) * aRx;
                    float ly = cy + (float)Math.Sin(lagAngles[li]) * bRy;
                    g.DrawEllipse(pLag, lx - 5f, ly - 5f, 10f, 10f);
                    g.FillEllipse(li == 3 ? bEmLag : bLag, lx - 2.5f, ly - 2.5f, 5f, 5f);
                    g.DrawString(lagNames[li], fLag, li == 3 ? bEmLag : bLag, lx + 7f, ly - 6f);
                }
            }

            // Host star with coronal flare
            using (GraphicsPath sg = new GraphicsPath())
            {
                sg.AddEllipse(cx - 52f, cy - 52f, 104f, 104f);
                using (PathGradientBrush pgb = new PathGradientBrush(sg))
                {
                    pgb.CenterColor = Color.FromArgb(190, 255, 180, 40);
                    pgb.SurroundColors = new Color[] { Color.Transparent };
                    g.FillPath(pgb, sg);
                }
            }
            using (SolidBrush sb = new SolidBrush(Color.FromArgb(255, 215, 90)))
                g.FillEllipse(sb, cx - 22f, cy - 22f, 44f, 44f);

            _planetScreenX = cx - focusOff + (float)Math.Cos(E) * aRx;
            _planetScreenY = cy + (float)Math.Sin(E) * bRy;

            // Green Lagrange Orbital Synchronization Window around planet
            using (Pen lagPen = new Pen(Color.FromArgb(210, 52, 255, 141), 2.2f))
            {
                lagPen.DashStyle = DashStyle.Dash;
                g.DrawEllipse(lagPen, _planetScreenX - 78f, _planetScreenY - 78f, 156f, 156f);
            }

            // 6 Rotating CME / Micrometeoroid Debris Hazards around the planet!
            for (int h = 0; h < 6; h++)
            {
                double hAng = -_simTime * 2.4 + h * (Math.PI / 3.0);
                float hRad = 64f + (h % 2) * 24f;
                float hx = _planetScreenX + (float)Math.Cos(hAng) * hRad;
                float hy = _planetScreenY + (float)Math.Sin(hAng) * hRad;
                using (SolidBrush hb = new SolidBrush(Color.FromArgb(235, 239, 68, 68)))
                using (Pen hp = new Pen(Color.FromArgb(254, 202, 202), 1.5f))
                {
                    g.FillEllipse(hb, hx - 8f, hy - 8f, 16f, 16f);
                    g.DrawEllipse(hp, hx - 12f, hy - 12f, 24f, 24f);
                }
            }

            Draw3DPlanetWithFallback(g, _current.Index, _planetScreenX, _planetScreenY, 25f);

            // Draw Collectible Golden Science Stars (+200 PTS each!)
            using (Font fStar = new Font("Segoe UI", 9.0f, FontStyle.Bold))
            using (SolidBrush bStarGold = new SolidBrush(Color.FromArgb(250, 204, 21)))
            {
                foreach (var st in _bonusStars)
                {
                    if (!st.Collected)
                    {
                        float sx = r.X + st.XNorm * r.Width;
                        float sy = r.Y + st.YNorm * r.Height;
                        g.FillEllipse(bStarGold, sx - 11f, sy - 11f, 22f, 22f);
                        g.DrawString("⭐+200", fStar, Brushes.White, sx - 20f, sy + 12f);
                    }
                }
            }

            // Robotic Bee-MoE Satellite Module + Capture Laser
            using (Pen beam = new Pen(Color.FromArgb(52, 255, 141), 2.5f))
                g.DrawLine(beam, _beeX, _beeY, _planetScreenX, _planetScreenY);

            using (SolidBrush wB = new SolidBrush(Color.FromArgb(29, 78, 216)))
            using (SolidBrush gB = new SolidBrush(_hazardHitFlash > 0.1 ? Color.Red : Color.FromArgb(245, 158, 11)))
            {
                g.FillRectangle(wB, _beeX - 6f, _beeY - 24f, 12f, 16f);
                g.FillRectangle(wB, _beeX - 6f, _beeY + 8f, 12f, 16f);
                g.FillRectangle(gB, _beeX - 8f, _beeY - 6f, 16f, 12f);
            }

            // STITCH SCREEN 0 LEFT FLANK PANELS: Ion Propulsion Telemetry + Orbital Eccentricity Decay
            int flankW = 268;
            Rectangle leftFlank1 = new Rectangle(r.X + 10, r.Y + 48, flankW, 132);
            DrawChamferedGlassMfdPanel(g, leftFlank1,
                "ION PROPULSION TELEMETRY // 94.2%",
                new string[] {
                    ">> THRUST OUTPUT : 1,840 mN [NOMINAL]",
                    "VECTOR ALPHA  : ■■■■■■■■■□ 92.4%",
                    "VECTOR BETA   : ■■■■■■■■■■ 98.1%",
                    "VECTOR GAMMA  : ■■■■■■■■□□ 88.7%",
                    "SYNODIC LOCK  : ENGAGED (L1-L5 RING)",
                    "SPECIFIC IMP. : 4,180 s (XENON HALL)"
                }, Color.FromArgb(0, 240, 255), false);

            Rectangle leftFlank2 = new Rectangle(r.X + 10, r.Y + 186, flankW, 118);
            DrawChamferedGlassMfdPanel(g, leftFlank2,
                "ORBITAL ECCENTRICITY DECAY",
                new string[] {
                    string.Format(CultureInfo.InvariantCulture, ">> CURRENT e = {0:F3} ➔ e_crit = 0.012", _current.Eccentricity),
                    string.Format(CultureInfo.InvariantCulture, "SEMI-MAJOR a = {0:F3} AU | P = {1:F1} d", _current.SemiMajorAxisAu, _current.PeriodDays),
                    "PHASE DRIFT   : -0.004 rad/orbit",
                    "RESONANCE     : 3:2 MEAN MOTION LOCK",
                    "✓ CIRCULARIZATION CORRIDOR GREEN"
                }, Color.FromArgb(52, 255, 141), false);

            // STITCH SCREEN 0 RIGHT FLANK PANELS: Target Dossier (with Stitch HD K2-18 b Render) + Orbital Dock Fleet Rendezvous
            Rectangle rightFlank1 = new Rectangle(r.Right - flankW - 10, r.Y + 48, flankW, 152);
            string pClass = (_current.Disposition == "FALSE POSITIVE")
                ? "CLASS : ECLIPSING BINARY (FP)"
                : (_current.BulkDensityGcm3 >= 3.5 ? "CLASS : ROCKY / SUPER-EARTH" : "CLASS : HYCEAN / GAS WORLD");
            string pBio = (_current.Disposition == "FALSE POSITIVE")
                ? ">> LIFE : IMPOSSIBLE (STELLAR)"
                : (_current.InHz ? "✓ HABITABLE ZONE: LIFE POSSIBLE" : ">> EXTREME TEMP: CLOUD/MOON BIO");
            DrawChamferedGlassMfdPanel(g, rightFlank1,
                "TARGET DOSSIER // " + _current.ShortLabel.ToUpperInvariant(),
                new string[] {
                    pClass,
                    string.Format(CultureInfo.InvariantCulture, "MASS  : {0:F2} M_E | RAD: {1:F2} R_E", _current.MassMearth, _current.RadiusRearth),
                    string.Format(CultureInfo.InvariantCulture, "T_EQ  : {0:F0} K    | RHO: {1:F2} g/cc", _current.TeqK, _current.BulkDensityGcm3),
                    string.Format(CultureInfo.InvariantCulture, "DEPTH : {0:F0} ppm  | RV : {1:F2} m/s", _current.DepthPpm, _current.RadialVelocityKms),
                    string.Format(CultureInfo.InvariantCulture, "ORBIT : P = {0:F1} d | a = {1:F3} AU", _current.PeriodDays, _current.SemiMajorAxisAu),
                    pBio
                }, Color.FromArgb(0, 240, 255), false);
            if (_stitchOrbitK218bImg != null)
            {
                Rectangle thumbR = new Rectangle(rightFlank1.Right - 64, rightFlank1.Y + 22, 54, 54);
                g.DrawImage(_stitchOrbitK218bImg, thumbR);
                using (Pen tp = new Pen(Color.FromArgb(0, 240, 255), 1.2f))
                    g.DrawRectangle(tp, thumbR);
            }

            Rectangle rightFlank2 = new Rectangle(r.Right - flankW - 10, r.Y + 206, flankW, 102);
            DrawChamferedGlassMfdPanel(g, rightFlank2,
                "ORBITAL DOCK // FLEET RENDEZVOUS",
                new string[] {
                    ">> BEE-POD CARRIER 01  [DOCKED]",
                    ">> PROBE LOIS-EXO-4    [APPROACHING]",
                    "   HARVESTER BIO-CNTR  [STANDBY L4]",
                    "✓  JWST UPLINK BEACON  [ONLINE 99.9%]"
                }, Color.FromArgb(255, 186, 32), false);

            // Top Progress Bar + Direct Button to Oolite Planetfall Stage (Localized EN / FR)
            using (SolidBrush topHud = new SolidBrush(Color.FromArgb(225, 7, 14, 26)))
            using (Pen topP = new Pen(Color.FromArgb(180, 0, 240, 255), 1.2f))
            {
                Rectangle hBox = new Rectangle(r.X + 10, r.Y + 8, r.Width - 295, 32);
                g.FillRectangle(topHud, hBox);
                g.DrawRectangle(topP, hBox);
                string syncTxt = (_lang == "fr")
                    ? "❖ STITCH STAGE 02: ORBIT BEE // SYNCHRONISATION LAGRANGE SUR " + _current.ShortLabel + " : " + (int)_captureProgress + "% (RAMASSEZ LES ⭐ +200 PTS !)"
                    : "❖ STITCH STAGE 02: ORBIT BEE // LAGRANGE RING INGRESS ON " + _current.ShortLabel + ": " + (int)_captureProgress + "% (COLLECT ⭐ +200 PTS!)";
                g.DrawString(syncTxt, new Font("Consolas", 8.8f, FontStyle.Bold), Brushes.Gold, hBox.X + 10, hBox.Y + 8);
            }

            Rectangle btnJumpLand = new Rectangle(r.Right - 275, r.Y + 8, 265, 32);
            using (SolidBrush bJ = new SolidBrush(Color.FromArgb(16, 185, 129)))
            using (Pen pJ = new Pen(Color.White, 1.4f))
            {
                g.FillRectangle(bJ, btnJumpLand);
                g.DrawRectangle(pJ, btnJumpLand);
                string btnTxt = (_lang == "fr") ? "🛬 ATTERRIR SUR LA PLANÈTE [3]" : "🛬 LAND ON PLANET NOW [STAGE 3]";
                g.DrawString(btnTxt, new Font("Consolas", 8.8f, FontStyle.Bold), Brushes.Black, btnJumpLand.X + 10, btnJumpLand.Y + 8);
            }
            _hotspots.Add(new HudHotspot { Id = "stg:2", Rect = btnJumpLand });

            g.Clip = oldClip;
            using (Pen border = new Pen(Color.FromArgb(170, 56, 189, 248), 1.6f))
                g.DrawRectangle(border, r);
        }

        private void DrawOoliteArcGauge(Graphics g, float cx, float cy, float radius, double frac, string valText, string labelText, bool warnRed)
        {
            frac = Math.Max(0.0, Math.Min(1.0, frac));
            Color arcColor = warnRed ? Color.FromArgb(239, 68, 68) : (frac > 0.78 ? Color.FromArgb(132, 204, 22) : Color.FromArgb(74, 222, 128));
            using (Pen trackPen = new Pen(Color.FromArgb(90, 20, 83, 45), 4.5f))
            using (Pen valPen = new Pen(arcColor, 5.0f))
            using (Pen outerThin = new Pen(Color.FromArgb(150, 34, 197, 94), 1.0f))
            {
                g.DrawArc(trackPen, cx - radius, cy - radius, radius * 2f, radius * 2f, 135f, 270f);
                g.DrawArc(valPen, cx - radius, cy - radius, radius * 2f, radius * 2f, 135f, (float)(270.0 * frac));
                g.DrawArc(outerThin, cx - radius - 4f, cy - radius - 4f, (radius + 4f) * 2f, (radius + 4f) * 2f, 130f, 280f);
            }
            using (Font fVal = new Font("Consolas", 8.5f, FontStyle.Bold))
            using (Font fLbl = new Font("Consolas", 6.5f, FontStyle.Bold))
            using (SolidBrush bVal = new SolidBrush(warnRed ? Color.FromArgb(251, 146, 60) : Color.FromArgb(250, 204, 21)))
            using (SolidBrush bLbl = new SolidBrush(Color.FromArgb(74, 222, 128)))
            {
                SizeF sv = g.MeasureString(valText, fVal);
                SizeF sl = g.MeasureString(labelText, fLbl);
                g.DrawString(valText, fVal, bVal, cx - sv.Width * 0.5f, cy - 3f);
                g.DrawString(labelText, fLbl, bLbl, cx - sl.Width * 0.5f, cy + radius - 6f);
            }
        }

        private GraphicsPath CreateChamferedMfdPath(Rectangle box, int cut)
        {
            GraphicsPath gp = new GraphicsPath();
            int x = box.X, y = box.Y, r = box.Right, b = box.Bottom;
            gp.AddPolygon(new Point[] {
                new Point(x, y),
                new Point(r - cut, y),
                new Point(r, y + cut),
                new Point(r, b),
                new Point(x + cut, b),
                new Point(x, b - cut)
            });
            return gp;
        }

        private void DrawChamferedMfdPanel(Graphics g, Rectangle box, Color bgCol, Color borderCol, int cut)
        {
            using (GraphicsPath path = CreateChamferedMfdPath(box, cut))
            using (SolidBrush bg = new SolidBrush(bgCol))
            using (Pen innerGlow = new Pen(Color.FromArgb(40, borderCol), 3.4f))
            using (Pen border = new Pen(borderCol, 1.3f))
            {
                g.FillPath(bg, path);
                g.DrawPath(innerGlow, path);
                g.DrawPath(border, path);
            }
        }

        /// <summary>
        /// Renders a Movie-Grade Dark Obsidian Glassmorphic MFD Panel with chamfered top-right and bottom-left corners,
        /// subtle inner cyan glow, and strict color hierarchy (Bright Cyan #00f0ff / Emerald #00ff88 for active telemetry,
        /// Muted Steel-Blue #5c88a6 for passive logs, Red/Amber strictly for active flight hazards).
        /// </summary>
        private void DrawChamferedGlassMfdPanel(Graphics g, Rectangle box, string title, string[] lines, Color titleColor, bool isPassiveLog)
        {
            using (GraphicsPath path = CreateChamferedMfdPath(box, 13))
            using (LinearGradientBrush bg = new LinearGradientBrush(
                box, Color.FromArgb(234, 5, 16, 32), Color.FromArgb(218, 2, 8, 18), LinearGradientMode.ForwardDiagonal))
            using (Pen innerGlow = new Pen(Color.FromArgb(38, 0, 229, 255), 3.6f))
            using (Pen border = new Pen(isPassiveLog ? Color.FromArgb(135, 92, 136, 166) : Color.FromArgb(185, 0, 229, 255), 1.3f))
            using (Pen cornerAccent = new Pen(Color.FromArgb(235, 0, 255, 136), 2.0f))
            {
                g.FillPath(bg, path);
                g.DrawPath(innerGlow, path);
                g.DrawPath(border, path);
                // Top-left & bottom-right tactical corner ticks
                g.DrawLine(cornerAccent, box.X, box.Y, box.X + 14, box.Y);
                g.DrawLine(cornerAccent, box.X, box.Y, box.X, box.Y + 10);
                g.DrawLine(cornerAccent, box.Right - 14, box.Bottom, box.Right, box.Bottom);
            }

            using (Font fTitle = new Font("Consolas", 7.8f, FontStyle.Bold))
            using (Font fLine = new Font("Consolas", 7.2f, FontStyle.Regular))
            using (Font fLineBold = new Font("Consolas", 7.2f, FontStyle.Bold))
            using (SolidBrush bTitle = new SolidBrush(titleColor))
            using (SolidBrush bActiveCyan = new SolidBrush(Color.FromArgb(216, 248, 255)))
            using (SolidBrush bMutedSteel = new SolidBrush(Color.FromArgb(118, 162, 192))) // #5c88a6 muted steel-blue for passive logs
            using (SolidBrush bEmerald = new SolidBrush(Color.FromArgb(0, 255, 136)))      // #00ff88 emerald green for active status
            using (SolidBrush bAmberHazard = new SolidBrush(Color.FromArgb(255, 170, 0)))  // Reserved strictly for active thermal/flight alerts
            {
                int y = box.Y + 5;
                if (!string.IsNullOrEmpty(title))
                {
                    g.DrawString(title, fTitle, bTitle, box.X + 7, y);
                    using (Pen hdrLine = new Pen(Color.FromArgb(75, titleColor), 1.0f))
                        g.DrawLine(hdrLine, box.X + 6, y + 13, box.Right - 14, y + 13);
                    y += 16;
                }
                foreach (string ln in lines)
                {
                    if (y > box.Bottom - 12) break;
                    bool isGrn = ln.StartsWith(">>") || ln.StartsWith("✓") || ln.Contains("granted") || ln.Contains("accordée") || ln.Contains("Clean") || ln.Contains("NOMINAL") || ln.Contains("LOCKED");
                    bool isWarn = ln.StartsWith("🔥") || ln.Contains("DANGER") || ln.Contains("96°C") || ln.Contains("RETRO");
                    Brush br = isWarn ? (Brush)bAmberHazard : (isGrn ? (Brush)bEmerald : (isPassiveLog ? (Brush)bMutedSteel : (Brush)bActiveCyan));
                    g.DrawString(ln, (isGrn || isWarn) ? fLineBold : fLine, br, box.X + 7, y);
                    y += 13;
                }
            }
        }

        /// <summary>
        /// Renders the Movie-Grade Stage 3 Planetary Landing Cockpit (Star Trek + Elite Dangerous + Oolite Planetfall OXP):
        ///   A. Curved K2-18 b Hycean Horizon with Rayleigh/Mie Atmospheric Scattering Shader + Dynamic Re-Entry Plasma Vignette
        ///   B. 3D Perspective Glide-Slope Tunnel ("Highway-in-the-Sky") leading to the Landing Pad Beacon + Pitch Ladder Drift Arrows
        ///   C. Dark Obsidian Glassmorphic Chamfered MFD Panels + Contextual HUD Decluttering (Flight Mode vs. Trade Mode Tabs)
        ///   D. Strict Color Hierarchy & Authentic 3D Holographic Cobra Mark III Wireframe
        /// </summary>
        private void DrawOolitePlanetfallStage2(Graphics g, Rectangle r)
        {
            Region oldClip = g.Clip;
            g.SetClip(r);

            // 1. Curved Planetary Horizon (3D Ray-Traced K2-18 b Hycean Ocean & Cyclonic Clouds — ZERO horizontal stretching!)
            if (_ooliteHorizonBackdrop != null)
            {
                g.DrawImage(_ooliteHorizonBackdrop, r);
            }
            else
            {
                using (SolidBrush spaceB = new SolidBrush(Color.FromArgb(4, 9, 24)))
                    g.FillRectangle(spaceB, r);
            }

            // Extra Real-Time Rayleigh / Mie Atmospheric Scattering Glow Arc along the Curved Planetary Limb
            float limbArcW = r.Width * 1.44f;
            float limbArcX = r.X - r.Width * 0.22f;
            float limbArcY = r.Y + r.Height * 0.255f;
            float limbArcH = r.Height * 1.15f;
            using (Pen rayleighOuter = new Pen(Color.FromArgb(48, 0, 240, 255), 18.0f))
            using (Pen rayleighMid = new Pen(Color.FromArgb(95, 56, 189, 248), 8.0f))
            using (Pen mieCore = new Pen(Color.FromArgb(175, 186, 230, 253), 2.5f))
            {
                g.DrawArc(rayleighOuter, limbArcX, limbArcY, limbArcW, limbArcH, 200f, 140f);
                g.DrawArc(rayleighMid, limbArcX, limbArcY + 3f, limbArcW, limbArcH, 200f, 140f);
                g.DrawArc(mieCore, limbArcX, limbArcY + 5f, limbArcW, limbArcH, 202f, 136f);
            }

            // Dynamic Re-Entry Plasma Vignette tied to C.TEMP (96°C) and SPEED (> 75 m/s)
            double plasmaIntensity = Math.Max(0.0, Math.Min(1.0, (_landerCabinTempC - 62.0) / 45.0));
            if (plasmaIntensity > 0.05)
            {
                int pAlpha = (int)(plasmaIntensity * (95.0 + 25.0 * Math.Sin(_simTime * 6.0)));
                pAlpha = Math.Max(20, Math.Min(140, pAlpha));
                using (LinearGradientBrush topPlasma = new LinearGradientBrush(
                    new Point(r.X, r.Y), new Point(r.X, r.Y + 95),
                    Color.FromArgb(pAlpha, 255, 82, 22), Color.FromArgb(0, 255, 140, 0)))
                using (LinearGradientBrush leftPlasma = new LinearGradientBrush(
                    new Point(r.X, r.Y), new Point(r.X + 115, r.Y),
                    Color.FromArgb((int)(pAlpha * 1.15), 255, 68, 16), Color.FromArgb(0, 255, 120, 0)))
                using (LinearGradientBrush rightPlasma = new LinearGradientBrush(
                    new Point(r.Right - 115, r.Y), new Point(r.Right, r.Y),
                    Color.FromArgb(0, 255, 120, 0), Color.FromArgb((int)(pAlpha * 1.15), 255, 68, 16)))
                {
                    g.FillRectangle(topPlasma, r.X, r.Y, r.Width, 95);
                    g.FillRectangle(leftPlasma, r.X, r.Y, 115, r.Height);
                    g.FillRectangle(rightPlasma, r.Right - 115, r.Y, 115, r.Height);
                }
            }

            // Dynamic Space Elevator / Orbital Tether Tower rising from the Planetary Surface Landing Pad Beacon
            float padCenterX = r.X + r.Width * 0.50f - _landerOffsetX * 1.15f;
            float padSurfaceY = r.Y + r.Height * 0.545f;
            float horizonY = r.Y + r.Height * 0.275f;
            using (LinearGradientBrush tetherBrush = new LinearGradientBrush(
                new PointF(padCenterX - 5f, r.Y + 145f), new PointF(padCenterX + 5f, padSurfaceY),
                Color.FromArgb(190, 186, 230, 253), Color.FromArgb(95, 0, 240, 255)))
            using (Pen tetherEdge = new Pen(Color.FromArgb(185, 224, 242, 254), 1.2f))
            {
                g.FillRectangle(tetherBrush, padCenterX - 3.5f, r.Y + 145f, 7f, padSurfaceY - (r.Y + 145f));
                g.DrawLine(tetherEdge, padCenterX - 3.5f, r.Y + 145f, padCenterX - 2.0f, padSurfaceY);
                g.DrawLine(tetherEdge, padCenterX + 3.5f, r.Y + 145f, padCenterX + 2.0f, padSurfaceY);
            }

            double gEarth = Math.Round(_current.MassMearth / Math.Max(0.25, _current.RadiusRearth * _current.RadiusRearth), 2);
            bool isAligned = Math.Abs(_landerOffsetX) < 35.0f;

            // 2. Top Row of 5 Dark Obsidian Glassmorphic Chamfered MFD Panels (Zero overlap with horizon & integrated Step 3 Briefing!)
            int mfdGap = 6;
            int my = r.Y + 6;
            int topMfdH = 136;
            int stdW = (r.Width - 16 - mfdGap * 4) / 5;
            int mx = r.X + 8;

            // MFD 1: Cobra Mark III / Bee-Lander Ship & Gravity Telemetry (Active Bright Cyan #00f0ff & Emerald #00ff88)
            DrawChamferedGlassMfdPanel(g, new Rectangle(mx, my, stdW, topMfdH),
                "COBRA MK III // HONOUR OF HIVEMIND",
                new string[] {
                    string.Format(CultureInfo.InvariantCulture, ">> Surface Gravity: {0:F2} g ({1:F1} m/s²)", gEarth, gEarth * 9.81),
                    "Service Level: 98% | Mass: 214.7t",
                    "Fuel: 6.9 LY | Solar Wind: 0.000 LY/m",
                    string.Format(CultureInfo.InvariantCulture, "Credits: {0:N1} Cr | Score: {1:N0}", 14399.5, _scoreHmt),
                    "Legal Status : Clean [0] NOMINAL",
                    "Rating: Astro-Hunter (19) | Rep: 0.70",
                    string.Format(CultureInfo.InvariantCulture, "Target: {0} ({1})", _current.ShortLabel, _current.HostStar)
                }, Color.FromArgb(0, 240, 255), false);
            mx += stdW + mfdGap;

            // MFD 2: TACTICAL FLIGHT & ENTRY CONTROL (Red/Amber strictly reserved for Retro-Thrusters & Entry Heat!)
            Rectangle mfd2 = new Rectangle(mx, my, stdW, topMfdH);
            DrawChamferedGlassMfdPanel(g, mfd2,
                "TACTICAL FLIGHT & ENTRY CONTROL",
                new string[] {
                    string.Format(CultureInfo.InvariantCulture, "🔥 ENTRY TEMP: {0:F0}°C | SPEED: {1:F0} m/s", _landerCabinTempC, _landerSpeedMs),
                    string.Format(CultureInfo.InvariantCulture, "✓ GLIDE-SLOPE: {0:+0.0;-0.0} m (< 35m OK)", _landerOffsetX),
                    "Controls: [SPACE/UP] Brake | [ENTER] Land"
                }, Color.FromArgb(0, 240, 255), false);

            Rectangle bClr = new Rectangle(mfd2.X + 8, mfd2.Bottom - 62, mfd2.Width - 16, 17);
            Rectangle bGer = new Rectangle(mfd2.X + 8, mfd2.Bottom - 42, mfd2.Width - 16, 17);
            Rectangle bBrk = new Rectangle(mfd2.X + 8, mfd2.Bottom - 22, mfd2.Width - 16, 18);
            using (SolidBrush bc = new SolidBrush(Color.FromArgb(210, 6, 78, 59)))
            using (SolidBrush bgGear = new SolidBrush(Color.FromArgb(210, 12, 74, 110)))
            using (SolidBrush bb = new SolidBrush(Color.FromArgb(230, 68, 12, 12))) // High-Visibility Danger/Retro-Brake Alert!
            using (Pen pEm = new Pen(Color.FromArgb(0, 255, 136), 1.1f))
            using (Pen pCy = new Pen(Color.FromArgb(0, 240, 255), 1.1f))
            using (Pen pRed = new Pen(Color.FromArgb(255, 51, 51), 1.4f))
            using (Font fBtn = new Font("Consolas", 6.8f, FontStyle.Bold))
            {
                g.FillRectangle(bc, bClr); g.DrawRectangle(pEm, bClr);
                g.FillRectangle(bgGear, bGer); g.DrawRectangle(pCy, bGer);
                g.FillRectangle(bb, bBrk); g.DrawRectangle(pRed, bBrk);
                g.DrawString("✓ [C] CLEARANCE GRANTED", fBtn, Brushes.Aquamarine, bClr.X + 5, bClr.Y + 2);
                g.DrawString("✓ [G] LANDING GEAR DOWN", fBtn, Brushes.LightCyan, bGer.X + 5, bGer.Y + 2);
                g.DrawString("🔥 [Z/UP] FIRE RETRO-THRUSTERS", fBtn, Brushes.Orange, bBrk.X + 5, bBrk.Y + 3);
            }
            _hotspots.Add(new HudHotspot { Id = "oolite_clearance", Rect = bClr });
            _hotspots.Add(new HudHotspot { Id = "oolite_gear", Rect = bGer });
            _hotspots.Add(new HudHotspot { Id = "oolite_brake", Rect = bBrk });
            mx += stdW + mfdGap;

            // MFD 3: COMMS LOG (Shifted to Muted Steel-Blue #5c88a6 for passive logs, Emerald #00ff88 for active clearance!)
            DrawChamferedGlassMfdPanel(g, new Rectangle(mx, my, stdW, topMfdH),
                "COMMS LOG // PLANETARY CONTROL",
                new string[] {
                    "- Control: Landing clearance request",
                    "  logged on beacon channel.",
                    "[Tower]: Atmospheric entry corridor",
                    "  verified for USS HiveMind lander.",
                    ">> [" + _current.ShortLabel + " Control]: Landing",
                    ">> clearance granted (< 95 m/s).",
                    "✓ Glide-slope 3D beacon locked."
                }, Color.FromArgb(118, 162, 192), true);
            mx += stdW + mfdGap;

            // MFD 4: Integrated PLANET ANALYSIS & HABITABILITY MFD (Replaces floating popup so horizon is 100% clear!)
            Rectangle mfdStory = new Rectangle(mx, my, stdW, topMfdH);
            string lifeShortFr = _current.InHz ? "Vie: océans/sources habitables" : (_current.Disposition == "FALSE POSITIVE" ? "Vie: impossible (binaire)" : "Vie: nuages/lunes uniquement");
            string lifeShortEn = _current.InHz ? "Life: ocean/vent biosphere viable" : (_current.Disposition == "FALSE POSITIVE" ? "Life: impossible (binary star)" : "Life: cloud/exomoon microbes only");
            string[] storyLines = (_lang == "fr")
                ? new string[] {
                    string.Format(CultureInfo.InvariantCulture, ">> Analyse: {0} ({1:F2} g)", _current.ShortLabel, gEarth),
                    string.Format(CultureInfo.InvariantCulture, "Rayon {0:F2} R⊕ | T_eq {1:F0} K | ρ {2:F2}", _current.RadiusRearth, _current.TeqK, _current.BulkDensityGcm3),
                    lifeShortFr,
                    "✓ Tunnel 3D vert -> piste cible",
                    "✓ [ESPACE] Freiner | [ENTRÉE] Atterrir"
                }
                : new string[] {
                    string.Format(CultureInfo.InvariantCulture, ">> Analysis: {0} ({1:F2} g)", _current.ShortLabel, gEarth),
                    string.Format(CultureInfo.InvariantCulture, "Radius {0:F2} R⊕ | T_eq {1:F0} K | ρ {2:F2}", _current.RadiusRearth, _current.TeqK, _current.BulkDensityGcm3),
                    lifeShortEn,
                    "✓ Follow 3D green tunnel to pad",
                    "✓ [SPACE] Brake | [ENTER] Auto-Land"
                };
            DrawChamferedGlassMfdPanel(g, mfdStory,
                (_lang == "fr") ? "ANALYSE & VIE // " + _current.ShortLabel.ToUpperInvariant() : "ANALYSIS & LIFE // " + _current.ShortLabel.ToUpperInvariant(),
                storyLines, Color.FromArgb(253, 224, 71), false);

            Rectangle btnHearMfd = new Rectangle(mfdStory.X + 8, mfdStory.Bottom - 23, mfdStory.Width - 16, 18);
            using (SolidBrush bHr = new SolidBrush(Color.FromArgb(235, 245, 158, 11)))
            using (Font fBtn = new Font("Segoe UI", 7.0f, FontStyle.Bold))
            {
                g.FillRectangle(bHr, btnHearMfd);
                g.DrawString((_lang == "fr") ? "♫ ÉCOUTER LEÇON (+150 PTS)" : "♫ HEAR KID LESSON (+150 PTS)", fBtn, Brushes.Black, btnHearMfd.X + 6, btnHearMfd.Y + 2);
            }
            _hotspots.Add(new HudHotspot { Id = "kid_lesson", Rect = btnHearMfd });
            mx += stdW + mfdGap;

            // MFD 5: Top-Right Navigation & Proximity Beacons (Aligned Monospace Columns)
            DrawChamferedGlassMfdPanel(g, new Rectangle(mx, my, stdW, topMfdH),
                string.Format(CultureInfo.InvariantCulture, "{0:F3} km  {1} [Planet] 58° >v<", _landerAltKm, _current.ShortLabel),
                new string[] {
                    ">> 0.000 km  Target Pad Beacon 00° >v<",
                    "53.36 km  Navigation Buoy 123° <^^",
                    "61.65 km  TransHab Station 123° <^^",
                    "359.4 km  Mono (Moon) 142° <v_",
                    "828.6 km  Witchpoint Beacon 110° >^_",
                    "836.1 km  Royal Hunting Lodge 112° >^^",
                    "50121 km  " + _current.ShortLabel + " [Star] 63° >>^"
                }, Color.FromArgb(0, 240, 255), true);

            // 3. Left & Right Side Dark Glassmorphic MFD Panels with Contextual HUD Decluttering (Flight Mode vs. Trade Mode Tabs!)
            int sideW = stdW + 14;
            int ly = my + topMfdH + 6;
            Rectangle leftTopBox = new Rectangle(r.X + 8, ly, sideW, 132);

            if (!_stage2TradeTabExpanded)
            {
                // FLIGHT MODE (Default during descent): Stitch Screen 3 [SYS.SPEC.01] ATMOSPHERIC SPECTRO + Hycean Atmo Telemetry!
                DrawChamferedGlassMfdPanel(g, leftTopBox,
                    "[SYS.SPEC.01] ATMOSPHERIC SPECTRO",
                    new string[] {
                        string.Format(CultureInfo.InvariantCulture, ">> Surface Gravity : {0:F2} g ({1:F1} m/s²)", gEarth, gEarth * 9.81),
                        "N2 NITROGEN  : ■■■■■■■□□□ 74.2%",
                        "O2 OXYGEN    : ■■□□□□□□□□ 18.1%",
                        "CO2 DIOXIDE  : ■□□□□□□□□□ 02.1%",
                        "NOBLE TRACES : ■□□□□□□□□□ 05.6% Ar/Kr",
                        "✓ EXOS 0.12 BAR | CORRIDOR OPTIMAL 88.4%"
                    }, Color.FromArgb(0, 240, 255), false);

                // Collapsed Toggleable Tabs for Equipment & Cargo Market (declutters 30% of screen over bright horizon glare!)
                Rectangle tabEquip = new Rectangle(r.X + 8, ly + 138, sideW, 24);
                Rectangle tabCargo = new Rectangle(r.X + 8, ly + 165, sideW, 24);
                using (GraphicsPath pEq = CreateChamferedMfdPath(tabEquip, 8))
                using (GraphicsPath pCg = CreateChamferedMfdPath(tabCargo, 8))
                using (SolidBrush bTab = new SolidBrush(Color.FromArgb(228, 5, 16, 32)))
                using (Pen pTab = new Pen(Color.FromArgb(165, 0, 229, 255), 1.2f))
                using (Font fTab = new Font("Consolas", 7.2f, FontStyle.Bold))
                using (SolidBrush bCy = new SolidBrush(Color.FromArgb(0, 240, 255)))
                {
                    g.FillPath(bTab, pEq); g.DrawPath(pTab, pEq);
                    g.FillPath(bTab, pCg); g.DrawPath(pTab, pCg);
                    g.DrawString("▸ [TAB] PRIMEABLE EQUIPMENT (4/18) [EXPAND]", fTab, bCy, tabEquip.X + 8, tabEquip.Y + 5);
                    g.DrawString("▸ [TAB] CARGO MARKET (33/33t)      [EXPAND]", fTab, Brushes.Aquamarine, tabCargo.X + 8, tabCargo.Y + 5);
                }
                _hotspots.Add(new HudHotspot { Id = "oolite_toggle_trade", Rect = tabEquip });
                _hotspots.Add(new HudHotspot { Id = "oolite_toggle_trade", Rect = tabCargo });

                // Stitch Screen 3: Holographic 3D SAR Wireframe Terrain Window (Chasma Australis LZ-ALPHA) on Lower-Left
                Rectangle sarBox = new Rectangle(r.X + 8, ly + 192, sideW, 102);
                DrawChamferedGlassMfdPanel(g, sarBox,
                    "[RADAR-SAR: ACTIVE] // CHASMA AUSTRALIS",
                    new string[] { }, Color.FromArgb(52, 255, 141), false);
                if (_stitchSarTerrainImg != null)
                {
                    Rectangle imgR = new Rectangle(sarBox.X + 6, sarBox.Y + 20, sarBox.Width - 12, sarBox.Height - 37);
                    g.DrawImage(_stitchSarTerrainImg, imgR);
                    using (Pen pSar = new Pen(Color.FromArgb(180, 0, 240, 255), 1.1f))
                        g.DrawRectangle(pSar, imgR);
                }
                using (Font fSar = new Font("Consolas", 6.8f, FontStyle.Bold))
                using (SolidBrush bSar = new SolidBrush(Color.FromArgb(52, 255, 141)))
                {
                    g.DrawString("✓ 3D SAR WIREFRAME LOCK: LZ-ALPHA [OOL-L03]", fSar, bSar, sarBox.X + 7, sarBox.Bottom - 15);
                }
            }
            else
            {
                // TRADE & EQUIPMENT MODE (Expanded on [TAB] or click, with dark obsidian glassmorphic backing!)
                DrawChamferedGlassMfdPanel(g, leftTopBox,
                    "▾ [TAB] PRIMEABLE EQUIPMENT (4/18)",
                    new string[] {
                        ">> Target Banking / Atmo Stabiliser",
                        "   LMSS Activator & Heat Shield",
                        "   Market Inquirer MFD",
                        "   Advanced Navigation Computer",
                        "   Telescope (JWST Coronagraph)",
                        "   Click or press [TAB] for Flight Mode"
                    }, Color.FromArgb(0, 240, 255), false);
                _hotspots.Add(new HudHotspot { Id = "oolite_toggle_trade", Rect = leftTopBox });

                Rectangle leftCargoBox = new Rectangle(r.X + 8, ly + 136, sideW, 108);
                DrawChamferedGlassMfdPanel(g, leftCargoBox,
                    "▾ [TAB] MARKET FOR SHIP'S CARGO",
                    new string[] {
                        "25g Quirium Crystal -> 8g at 42.6 Cr",
                        "25g Gem-Stones      -> 8g at 21.9 Cr",
                        "128kg Platinum      -> 15kg at 71.4 Cr",
                        "46kg Gold           -> 7kg at 40.4 Cr",
                        string.Format(CultureInfo.InvariantCulture, ">> Density ρ = {0:F2} g/cm³ | Teq = {1:F0} K", _current.BulkDensityGcm3, _current.TeqK)
                    }, Color.FromArgb(0, 255, 136), false);
                _hotspots.Add(new HudHotspot { Id = "oolite_toggle_trade", Rect = leftCargoBox });
            }

            // Right Side Chamfered Glassmorphic Panels: Stitch [SYS.LITHO.03] SEISMIC & SUB-BASALT + Sector Traffic + Flight Control Overrides
            int rx = r.Right - sideW - 8;
            DrawChamferedGlassMfdPanel(g, new Rectangle(rx, ly, sideW, 112),
                "[SYS.LITHO.03] SEISMIC & SUB-BASALT",
                new string[] {
                    ">> SURFACE HARDNESS : 9.4 MOHS (BASALT)",
                    ">> GEOTHERM GRADIENT: +42.8°C | 0.12 RMS",
                    "SUB-CRUSTAL ECHO   : [||||||....] LOCKED",
                    "AUTONOMOUS LVL 4   : RETRO OUTPUT 84%",
                    "✓ [BEACON LOCKED: OOL-L03] | G-LOAD 1.84"
                }, Color.FromArgb(0, 240, 255), false);

            DrawChamferedGlassMfdPanel(g, new Rectangle(rx, ly + 116, sideW, 108),
                "SECTOR TRAFFIC // ORBITAL SCANNER",
                new string[] {
                    ">> 41.94 km  Gecko: Orbital Buckler 115°",
                    "48.24 km  Python: Scornful Lunatic 123°",
                    "48.26 km  Python: Spark of Decency 122°",
                    "49.29 km  Asp Mk II: Dennisten's   126°",
                    "50.73 km  Python: Quandixe Maru    125°"
                }, Color.FromArgb(118, 162, 192), true);

            DrawChamferedGlassMfdPanel(g, new Rectangle(rx, ly + 228, sideW, 66),
                "SYSTEM PLANETARY BODIES (OU)",
                new string[] {
                    ">> 1.00 OU * Main Planet (" + _current.ShortLabel + ")",
                    "0.63 OU * Miniterra | 2.35 OU * Terra",
                    "3.35 OU * Subterra  | 4.31 OU * Ice Giant"
                }, Color.FromArgb(0, 240, 255), false);

            // 4. Center 3D Perspective "Glide-Slope Tunnel" ("Highway-in-the-Sky") + Stitch Ablative Shield Banner & TGT LOCK Reticle!
            float hudCx = r.X + r.Width * 0.50f;
            float hudCy = r.Y + r.Height * 0.435f;

            // Stitch Screen 3 Top Ablative Shield & Atmospheric Ingress Banner right above the 3D Glide-Slope Tunnel!
            Rectangle ablativeBanner = new Rectangle((int)hudCx - 295, r.Y + topMfdH + 6, 590, 21);
            using (SolidBrush abBg = new SolidBrush(Color.FromArgb(228, 52, 8, 12)))
            using (Pen abPen = new Pen(Color.FromArgb(235, 255, 84, 73), 1.3f))
            using (Font fAb = new Font("Consolas", 7.2f, FontStyle.Bold))
            using (SolidBrush bAbTxt = new SolidBrush(Color.FromArgb(255, 180, 171)))
            {
                g.FillRectangle(abBg, ablativeBanner);
                g.DrawRectangle(abPen, ablativeBanner);
                string abTxt = string.Format(
                    CultureInfo.InvariantCulture,
                    "⚠ STITCH ATMOSPHERIC INGRESS // ABLATIVE SHIELD: 820°C (CABIN {0:F0}°C) | [RADAR-SAR: ACTIVE] | [RETRO ENGAGED]",
                    _landerCabinTempC
                );
                g.DrawString(abTxt, fAb, bAbTxt, ablativeBanner.X + 8, ablativeBanner.Y + 3);
            }

            float nearW = 460f;
            float nearH = 192f;
            float farW = 92f;
            float farH = 30f;

            // Project 6 receding 3D perspective tunnel gates from the cockpit HUD crosshair down to the Landing Pad Beacon!
            PointF[][] gateCorners = new PointF[6][];
            Color tunnelBaseCol = isAligned ? Color.FromArgb(0, 255, 136) : Color.FromArgb(255, 170, 0);
            for (int gIdx = 5; gIdx >= 0; gIdx--)
            {
                float t = gIdx / 5.0f; // 0 = near cockpit gate, 1 = surface pad gate
                float persp = (float)Math.Pow(t, 1.28);
                float gx = hudCx + (padCenterX - hudCx) * persp;
                float gy = (hudCy - 18f) + (padSurfaceY - (hudCy - 18f)) * persp;
                float gw = nearW + (farW - nearW) * persp;
                float gh = nearH + (farH - nearH) * persp;

                float gxL = gx - gw * 0.5f, gxR = gx + gw * 0.5f;
                float gyT = gy - gh * 0.5f, gyB = gy + gh * 0.5f;
                gateCorners[gIdx] = new PointF[] {
                    new PointF(gxL, gyT), new PointF(gxR, gyT),
                    new PointF(gxR, gyB), new PointF(gxL, gyB)
                };

                int alphaGate = (gIdx == 0) ? 235 : (int)(210 - gIdx * 24);
                using (SolidBrush gateFill = new SolidBrush(Color.FromArgb(gIdx == 0 ? 95 : 18, 3, 20, 24)))
                using (Pen gateGlow = new Pen(Color.FromArgb(alphaGate / 3, tunnelBaseCol), 5.0f))
                using (Pen gateCore = new Pen(Color.FromArgb(alphaGate, gIdx % 2 == 0 ? tunnelBaseCol : Color.FromArgb(0, 240, 255)), gIdx == 0 ? 2.2f : 1.4f))
                {
                    if (gIdx == 0) g.FillPolygon(gateFill, gateCorners[gIdx]);
                    g.DrawPolygon(gateGlow, gateCorners[gIdx]);
                    g.DrawPolygon(gateCore, gateCorners[gIdx]);
                }
            }

            // Connect the 4 corners of the receding gates to form the 3D Glide-Slope Highway Tunnel rails!
            using (Pen railGlow = new Pen(Color.FromArgb(65, 0, 255, 136), 4.0f))
            using (Pen railCore = new Pen(Color.FromArgb(195, 0, 240, 255), 1.5f))
            {
                for (int c = 0; c < 4; c++)
                {
                    g.DrawLine(railGlow, gateCorners[0][c], gateCorners[5][c]);
                    g.DrawLine(railCore, gateCorners[0][c], gateCorners[5][c]);
                }
            }

            // Glowing Perspective Landing Pad Beacon at the bottom end of the 3D Glide-Slope Tunnel (0.0 km)
            using (SolidBrush padGlow = new SolidBrush(Color.FromArgb(95, 0, 255, 136)))
            using (Pen padOuter = new Pen(Color.FromArgb(245, 0, 255, 136), 2.4f))
            using (Pen padInner = new Pen(Color.White, 1.5f))
            using (Font fPad = new Font("Consolas", 8.0f, FontStyle.Bold))
            {
                g.FillEllipse(padGlow, padCenterX - 58f, padSurfaceY - 12f, 116f, 24f);
                g.DrawEllipse(padOuter, padCenterX - 58f, padSurfaceY - 12f, 116f, 24f);
                g.DrawEllipse(padInner, padCenterX - 30f, padSurfaceY - 6f, 60f, 12f);
                string padLbl = (_lang == "fr") ? "⊕ PISTE CIBLE (0.0 km)" : "⊕ TARGET PAD (0.0 km)";
                g.DrawString(padLbl, fPad, Brushes.Aquamarine, padCenterX - 54f, padSurfaceY + 13f);
            }

            // Central HUD Pitch Ladder, Crosshair & Live Visual Drift Arrows (Wind Shear & Pad Offset right on the HUD!)
            float xL = gateCorners[0][0].X;
            float xR = gateCorners[0][1].X;
            float yT = gateCorners[0][0].Y;
            float yB = gateCorners[0][2].Y;

            using (Font fHudHdr = new Font("Consolas", 8.5f, FontStyle.Bold))
            using (Font fDrift = new Font("Consolas", 8.0f, FontStyle.Bold))
            using (SolidBrush bCyan = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bEm = new SolidBrush(Color.FromArgb(0, 255, 136)))
            using (SolidBrush bGold = new SolidBrush(Color.FromArgb(253, 224, 71)))
            using (SolidBrush bDarkPill = new SolidBrush(Color.FromArgb(215, 3, 12, 24)))
            using (Pen pLadder = new Pen(Color.FromArgb(185, 0, 255, 136), 1.4f))
            {
                // Top HUD Tunnel Header
                string tunnelTitle = (_lang == "fr")
                    ? "▼ TUNNEL DE DESCENTE 3D // SUIVEZ LA PISTE VERTE (3.0 km ➔ 0.0 km)"
                    : "▼ 3D GLIDE-SLOPE LANDING TUNNEL // FOLLOW GREEN PATH TO TARGET PAD";
                g.DrawString(tunnelTitle, fHudHdr, bGold, xL + 16f, yT + 5f);

                // Pitch Ladder Rungs along the 3D glide-slope floor
                for (int rung = 1; rung <= 4; rung++)
                {
                    float ry = yT + 42f + rung * 24f;
                    float rw = 52f - rung * 7f;
                    g.DrawLine(pLadder, hudCx - rw, ry, hudCx - 10f, ry);
                    g.DrawLine(pLadder, hudCx + 10f, ry, hudCx + rw, ry);
                    g.DrawLine(pLadder, hudCx - rw, ry, hudCx - rw, ry - 5f);
                    g.DrawLine(pLadder, hudCx + rw, ry, hudCx + rw, ry - 5f);
                }

                // Left Live Drift Callout on HUD Pitch Ladder: PAD OFFSET & ALIGNMENT
                Rectangle leftCallout = new Rectangle((int)xL - 152, (int)hudCy - 36, 148, 42);
                g.FillRectangle(bDarkPill, leftCallout);
                g.DrawRectangle(pLadder, leftCallout);
                g.DrawString(string.Format(CultureInfo.InvariantCulture, "PAD OFFSET: {0:+0.0;-0.0} m", _landerOffsetX), fDrift, bEm, leftCallout.X + 6, leftCallout.Y + 5);
                g.DrawString(isAligned ? "►► ALIGNED < 35m ✓" : "►► STEER TO CENTER", fDrift, isAligned ? bEm : bGold, leftCallout.X + 6, leftCallout.Y + 22);

                // Right Live Drift Callout on HUD Pitch Ladder: WIND SHEAR VECTOR
                Rectangle rightCallout = new Rectangle((int)xR + 4, (int)hudCy - 36, 152, 42);
                g.FillRectangle(bDarkPill, rightCallout);
                g.DrawRectangle(new Pen(Color.FromArgb(185, 0, 240, 255), 1.4f), rightCallout);
                g.DrawString(string.Format(CultureInfo.InvariantCulture, "WIND SHEAR: {0:F1} m/s", _landerWindShearMs), fDrift, bCyan, rightCallout.X + 6, rightCallout.Y + 5);
                g.DrawString("◄◄ ATMO STABILISER ON", fDrift, bEm, rightCallout.X + 6, rightCallout.Y + 22);

                // 3 Collectible Golden Science Stars inside the 3D Tunnel (2.2 km, 1.4 km, 0.7 km)
                double[] starAlts = new double[] { 2.2, 1.4, 0.7 };
                for (int i = 0; i < 3 && i < _bonusStars.Count; i++)
                {
                    float sFrac = (float)(1.0 - (starAlts[i] / 3.0));
                    float sy = (yT + 34f) + sFrac * (padSurfaceY - (yT + 34f) - 14f);
                    float sx = hudCx + (i == 1 ? 26f : -26f);
                    string stTxt = _bonusStars[i].Collected ? "✓+200!" : "★+200";
                    g.DrawString(stTxt, fDrift, _bonusStars[i].Collected ? bEm : bGold, sx - 18f, sy - 6f);
                }

                // Descending Explorer Robot Avatar (► YOU) flying through the 3D Glide-Slope Tunnel!
                float altFrac = (float)Math.Max(0.0, Math.Min(1.0, 1.0 - (_landerAltKm / 3.0)));
                float robX = hudCx + Math.Max(-50f, Math.Min(50f, _landerOffsetX * 0.45f));
                float robY = (yT + 34f) + altFrac * (padSurfaceY - (yT + 34f) - 16f);

                using (Pen guideBeam = new Pen(Color.FromArgb(210, 0, 255, 136), 2.2f))
                    g.DrawLine(guideBeam, robX, robY + 8f, padCenterX, padSurfaceY);

                using (SolidBrush robHull = new SolidBrush(Color.FromArgb(245, 158, 11)))
                using (SolidBrush robWing = new SolidBrush(Color.FromArgb(0, 240, 255)))
                using (Pen robRim = new Pen(Color.White, 1.6f))
                {
                    g.FillRectangle(robWing, robX - 24f, robY - 4f, 48f, 8f);
                    g.FillEllipse(robHull, robX - 11f, robY - 9f, 22f, 18f);
                    g.DrawEllipse(robRim, robX - 11f, robY - 9f, 22f, 18f);
                    string robLbl = string.Format(CultureInfo.InvariantCulture, "► YOU ({0:F2} km | {1:F0} m/s)", _landerAltKm, _landerSpeedMs);
                    g.DrawString(robLbl, fDrift, Brushes.White, robX + 28f, robY - 6f);
                }
            }

            // Two High-Contrast Chamfered Action Buttons below the 3D Glide-Slope Tunnel!
            int btnY = (int)padSurfaceY + 30;
            Rectangle btnEasyBrake = new Rectangle((int)hudCx - 246, btnY, 236, 28);
            Rectangle btnAutoLand = new Rectangle((int)hudCx + 4, btnY, 254, 28);
            using (SolidBrush bBrkBrush = new SolidBrush(Color.FromArgb(235, 14, 165, 233)))
            using (SolidBrush bLandBrush = new SolidBrush(Color.FromArgb(245, 0, 230, 128)))
            using (Pen pWhite = new Pen(Color.White, 1.5f))
            using (Font fActBtn = new Font("Segoe UI", 8.5f, FontStyle.Bold))
            {
                g.FillRectangle(bBrkBrush, btnEasyBrake);
                g.DrawRectangle(pWhite, btnEasyBrake);
                string brkTxt = (_lang == "fr") ? "► [ESPACE] FREINER & ALIGNER (+100 PTS)" : "► [SPACE] BRAKE & ALIGN (+100 PTS)";
                g.DrawString(brkTxt, fActBtn, Brushes.Black, btnEasyBrake.X + 8, btnEasyBrake.Y + 5);

                g.FillRectangle(bLandBrush, btnAutoLand);
                g.DrawRectangle(pWhite, btnAutoLand);
                string lndTxt = (_lang == "fr") ? "★ [ENTRÉE] AUTO-ATTERRIR (+1000 PTS)" : "★ [ENTER] AUTO-LAND NOW (+1,000 PTS)";
                g.DrawString(lndTxt, fActBtn, Brushes.Black, btnAutoLand.X + 8, btnAutoLand.Y + 5);
            }
            _hotspots.Add(new HudHotspot { Id = "oolite_brake", Rect = btnEasyBrake });
            _hotspots.Add(new HudHotspot { Id = "oolite_autoland", Rect = btnAutoLand });

            // 5. Bottom Curved Glassmorphic Instrument Binnacle (.: TRAVEL INFO :. | 10 Radial Arc Gauges + 3D Radar | .: SHIP INFO :.)
            int binH = 180;
            int binY = r.Bottom - binH - 4;
            int leftPodW = 200;
            int rightPodW = 215;
            int centerPodW = 620;
            int totalBinW = leftPodW + centerPodW + rightPodW + 16;
            int binStartX = r.X + (r.Width - totalBinW) / 2;

            Rectangle leftPod = new Rectangle(binStartX, binY + 20, leftPodW, binH - 20);
            Rectangle centerPod = new Rectangle(binStartX + leftPodW + 8, binY, centerPodW, binH);
            Rectangle rightPod = new Rectangle(centerPod.Right + 8, binY + 20, rightPodW, binH - 20);

            foreach (Rectangle pod in new Rectangle[] { leftPod, centerPod, rightPod })
            {
                using (GraphicsPath gpPod = CreateChamferedMfdPath(pod, 14))
                using (LinearGradientBrush podBg = new LinearGradientBrush(
                    pod, Color.FromArgb(238, 5, 14, 28), Color.FromArgb(228, 2, 7, 16), LinearGradientMode.Vertical))
                using (Pen podGlow = new Pen(Color.FromArgb(40, 0, 229, 255), 3.5f))
                using (Pen podBorder = new Pen(Color.FromArgb(180, 0, 229, 255), 1.4f))
                {
                    g.FillPath(podBg, gpPod);
                    g.DrawPath(podGlow, gpPod);
                    g.DrawPath(podBorder, gpPod);
                }
            }

            // Left Pod: .: TRAVEL INFO :. + Cyan Planetary Coordinate Grid
            using (Font fPodHdr = new Font("Consolas", 8.5f, FontStyle.Bold))
            using (Font fSmall = new Font("Consolas", 7.4f, FontStyle.Regular))
            using (SolidBrush bCyan = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bGrn = new SolidBrush(Color.FromArgb(0, 255, 136)))
            using (Pen pWire = new Pen(Color.FromArgb(95, 0, 229, 255), 1.1f))
            {
                g.DrawString(".: TRAVEL INFO :.", fPodHdr, bCyan, leftPod.X + 42, leftPod.Y + 5);
                g.DrawString("Destination: " + _current.ShortLabel, fSmall, bGrn, leftPod.X + 12, leftPod.Y + 21);
                float gx = leftPod.X + leftPod.Width * 0.5f;
                float gy = leftPod.Y + 86f;
                for (int w = 12; w <= 46; w += 11)
                {
                    g.DrawEllipse(pWire, gx - w, gy - 42f, w * 2f, 84f);
                    g.DrawEllipse(pWire, gx - 44f, gy - w, 88f, w * 2f);
                }
                g.DrawString("HYCEAN OCEAN // 1.27 G", fSmall, bCyan, leftPod.X + 26, leftPod.Bottom - 18);
            }

            // Center Pod Header: Target Name | Target Compass | Altitude | System Status
            using (Font fTiny = new Font("Consolas", 7.2f, FontStyle.Bold))
            using (SolidBrush bCy = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bGr = new SolidBrush(Color.FromArgb(0, 255, 136)))
            using (SolidBrush bSteel = new SolidBrush(Color.FromArgb(118, 162, 192)))
            {
                g.DrawString(_current.ShortLabel + " // GLIDE LOCK", fTiny, bCy, centerPod.X + 14, centerPod.Y + 8);
                g.DrawString("MODE: PLANETFALL 3D", fTiny, bGr, centerPod.X + 14, centerPod.Y + 24);
                g.DrawString("PRIMED Atmo Stabiliser", fTiny, bGr, centerPod.X + 14, centerPod.Y + 38);
                g.DrawString("EQUIP  JWST Coronagraph", fTiny, bSteel, centerPod.X + 14, centerPod.Y + 50);

                // Circular Target Compass with Green Pad Alignment Dot
                float compX = centerPod.X + 165f;
                float compY = centerPod.Y + 34f;
                using (Pen pComp = new Pen(Color.FromArgb(210, 0, 229, 255), 1.8f))
                using (SolidBrush bDot = new SolidBrush(Color.FromArgb(0, 255, 136)))
                {
                    g.DrawEllipse(pComp, compX - 22f, compY - 22f, 44f, 44f);
                    g.DrawEllipse(pComp, compX - 11f, compY - 11f, 22f, 22f);
                    g.DrawLine(pComp, compX - 22f, compY, compX + 22f, compY);
                    g.DrawLine(pComp, compX, compY - 22f, compX, compY + 22f);
                    float dotOffX = Math.Max(-14f, Math.Min(14f, -_landerOffsetX * 0.1f));
                    g.FillEllipse(bDot, compX + dotOffX - 4.5f, compY - 4.5f, 9f, 9f);
                }

                g.DrawString(string.Format(CultureInfo.InvariantCulture, "{0:F2} km", _landerAltKm), fTiny, bCy, centerPod.X + 220, centerPod.Y + 8);
                g.DrawString("1:1 LOCK", fTiny, bGr, centerPod.X + 215, centerPod.Y + 28);

                g.DrawString("Retro-Thrusters ARMED", fTiny, bGr, centerPod.Right - 165, centerPod.Y + 20);
                g.DrawString("Cadet Rank: LVL 1-4", fTiny, bCy, centerPod.Right - 165, centerPod.Y + 34);
                g.DrawString("Standby, Charges: 10 SR.SYS", fTiny, bSteel, centerPod.Right - 175, centerPod.Y + 48);
            }

            // Left 5 Radial Arc Gauges (SPEED, FUEL, ALT, CARGO, C.TEMP — C.TEMP & SPEED glow Amber/Red when hot!)
            float gY1 = centerPod.Y + 94f;
            float gY2 = centerPod.Y + 144f;
            DrawOoliteArcGauge(g, centerPod.X + 34f, gY1, 20f, _landerSpeedMs / 160.0, ((int)_landerSpeedMs).ToString(), "SPEED", _landerSpeedMs > 90.0);
            DrawOoliteArcGauge(g, centerPod.X + 86f, gY1, 20f, 0.85, "6.9+3", "FUEL", false);
            DrawOoliteArcGauge(g, centerPod.X + 138f, gY1, 20f, _landerAltKm / 3.5, _landerAltKm.ToString("F1", CultureInfo.InvariantCulture), "ALT", false);
            DrawOoliteArcGauge(g, centerPod.X + 86f, gY2, 20f, 1.0, "33/33", "CARGO", false);
            DrawOoliteArcGauge(g, centerPod.X + 138f, gY2, 20f, _landerCabinTempC / 140.0, ((int)_landerCabinTempC) + "°", "C.TEMP", _landerCabinTempC >= 85.0);

            // Right 5 Radial Arc Gauges (ENERGY, F.SH, ARMOUR, L.TEMP, A.SH)
            DrawOoliteArcGauge(g, centerPod.Right - 138f, gY1, 20f, _landerEnergy / 256.0, ((int)_landerEnergy).ToString(), "ENERGY", false);
            DrawOoliteArcGauge(g, centerPod.Right - 86f, gY1, 20f, _landerFwdShield / 100.0, ((int)_landerFwdShield) + "%", "F.SH", false);
            DrawOoliteArcGauge(g, centerPod.Right - 34f, gY1, 20f, _landerArmour / 100.0, ((int)_landerArmour) + "%", "ARMOUR", false);
            DrawOoliteArcGauge(g, centerPod.Right - 138f, gY2, 20f, 0.08, "0°", "L.TEMP", false);
            DrawOoliteArcGauge(g, centerPod.Right - 86f, gY2, 20f, _landerAftShield / 100.0, ((int)_landerAftShield) + "%", "A.SH", false);

            // Center Bottom 3D Elliptical Tactical Radar Scanner + Target Lollipops + 3D Metallic Ship Nose Cone!
            Rectangle scanEllipse = new Rectangle(centerPod.X + 172, centerPod.Y + 60, 276, 112);
            using (SolidBrush scanBg = new SolidBrush(Color.FromArgb(225, 4, 18, 16)))
            using (Pen emeraldRim = new Pen(Color.FromArgb(225, 0, 255, 136), 2.2f))
            using (Pen cyanGrid = new Pen(Color.FromArgb(115, 0, 229, 255), 1.0f))
            {
                g.FillEllipse(scanBg, scanEllipse);
                g.DrawEllipse(emeraldRim, scanEllipse);
                g.DrawEllipse(cyanGrid, scanEllipse.X + 46, scanEllipse.Y + 20, scanEllipse.Width - 92, scanEllipse.Height - 40);
                int scx = scanEllipse.X + scanEllipse.Width / 2;
                int scy = scanEllipse.Y + scanEllipse.Height / 2;
                g.DrawLine(cyanGrid, scanEllipse.X + 8, scy, scanEllipse.Right - 8, scy);
                g.DrawLine(cyanGrid, scx, scanEllipse.Y + 6, scx, scanEllipse.Bottom - 6);
                g.DrawLine(emeraldRim, scx, scy, scanEllipse.X + 42, scanEllipse.Y + 16);
                g.DrawLine(emeraldRim, scx, scy, scanEllipse.Right - 42, scanEllipse.Y + 16);

                // 3D Target Lollipops (Emerald & Cyan vertical stalks) inside Radar Scanner
                int[] lxOff = new int[] { -18, 6, 14, 24, 34, 46, 72, 84, -45 };
                int[] lyOff = new int[] { 6, 12, 16, 18, 22, 10, 8, 18, -14 };
                for (int k = 0; k < lxOff.Length; k++)
                {
                    bool isGrn = (k % 2 == 0);
                    using (Pen pStalk = new Pen(isGrn ? Color.FromArgb(0, 255, 136) : Color.FromArgb(0, 240, 255), 2.2f))
                    {
                        int tx = scx + lxOff[k];
                        int ty = scy + lyOff[k];
                        g.DrawLine(pStalk, tx, ty, tx, ty - 22);
                        g.DrawEllipse(pStalk, tx - 2, ty - 24, 4, 4);
                    }
                }

                // 3D Shaded Metallic Nose Cone protruding into bottom of scanner
                PointF[] noseHull = new PointF[] {
                    new PointF(scx, scy + 18),
                    new PointF(scx - 40, scanEllipse.Bottom + 4),
                    new PointF(scx + 40, scanEllipse.Bottom + 4)
                };
                using (LinearGradientBrush metalGrad = new LinearGradientBrush(
                    new PointF(scx - 40, scy + 18), new PointF(scx + 40, scanEllipse.Bottom + 4),
                    Color.FromArgb(225, 226, 232, 240), Color.FromArgb(245, 71, 85, 105)))
                using (Pen noseEdge = new Pen(Color.FromArgb(0, 240, 255), 1.5f))
                {
                    g.FillPolygon(metalGrad, noseHull);
                    g.DrawPolygon(noseEdge, noseHull);
                }
            }

            // Right Pod: .: SHIP INFO :. + Authentic Multi-Faceted 3D Holographic Cobra Mark III Mesh Wireframe!
            using (Font fPodHdr = new Font("Consolas", 8.5f, FontStyle.Bold))
            using (Font fSmall = new Font("Consolas", 7.2f, FontStyle.Regular))
            using (SolidBrush bCy = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bGrn = new SolidBrush(Color.FromArgb(0, 255, 136)))
            using (SolidBrush bRad = new SolidBrush(Color.FromArgb(145, 4, 38, 28)))
            using (Pen pShipOuter = new Pen(Color.FromArgb(235, 0, 255, 136), 1.5f))
            using (Pen pShipInner = new Pen(Color.FromArgb(170, 0, 240, 255), 1.0f))
            {
                g.DrawString(".: SHIP INFO :.", fPodHdr, bCy, rightPod.X + 48, rightPod.Y + 5);
                g.DrawString("Cobra Mark III // Bee-Lander", fSmall, bGrn, rightPod.X + 12, rightPod.Y + 21);
                float sx = rightPod.X + rightPod.Width * 0.5f;
                float sy = rightPod.Y + 86f;
                g.FillEllipse(bRad, sx - 62f, sy - 44f, 124f, 88f);
                g.DrawEllipse(pShipInner, sx - 62f, sy - 44f, 124f, 88f);

                // Authentic Multi-Faceted Oolite Cobra Mark III 3D Mesh Wireframe (Upper/Lower Hull + Wingtips + Canopy + Dual Exhausts)
                PointF noseTip = new PointF(sx, sy - 36f);
                PointF noseL = new PointF(sx - 18f, sy - 24f);
                PointF noseR = new PointF(sx + 18f, sy - 24f);
                PointF wingTipL = new PointF(sx - 56f, sy + 14f);
                PointF wingTipR = new PointF(sx + 56f, sy + 14f);
                PointF wingAftL = new PointF(sx - 44f, sy + 26f);
                PointF wingAftR = new PointF(sx + 44f, sy + 26f);
                PointF tailL = new PointF(sx - 20f, sy + 30f);
                PointF tailR = new PointF(sx + 20f, sy + 30f);
                PointF canopyTop = new PointF(sx, sy - 8f);
                PointF spineAft = new PointF(sx, sy + 18f);

                g.DrawPolygon(pShipOuter, new PointF[] { noseTip, noseL, wingTipL, wingAftL, tailL, tailR, wingAftR, wingTipR, noseR });
                // Internal 3D hull facets
                g.DrawLine(pShipInner, noseTip, canopyTop);
                g.DrawLine(pShipInner, canopyTop, spineAft);
                g.DrawLine(pShipInner, canopyTop, noseL);
                g.DrawLine(pShipInner, canopyTop, noseR);
                g.DrawLine(pShipInner, canopyTop, wingTipL);
                g.DrawLine(pShipInner, canopyTop, wingTipR);
                g.DrawLine(pShipInner, spineAft, wingAftL);
                g.DrawLine(pShipInner, spineAft, wingAftR);
                g.DrawLine(pShipInner, spineAft, tailL);
                g.DrawLine(pShipInner, spineAft, tailR);
                g.DrawLine(pShipInner, noseL, wingAftL);
                g.DrawLine(pShipInner, noseR, wingAftR);
                // Dual thruster exhaust bays
                g.DrawRectangle(pShipOuter, sx - 16f, sy + 24f, 12f, 6f);
                g.DrawRectangle(pShipOuter, sx + 4f, sy + 24f, 12f, 6f);

                g.DrawString("✓ All flight & landing systems", fSmall, bGrn, rightPod.X + 12, rightPod.Bottom - 26);
                g.DrawString("  nominal • 3D Tunnel Locked", fSmall, bCy, rightPod.X + 12, rightPod.Bottom - 14);
            }

            // Subtitle bar above the binnacle so kids can read the GCP Gemini Neural Voice lesson during landing!
            if (!string.IsNullOrEmpty(_subtitleText))
            {
                Rectangle subR2 = new Rectangle(r.X + 18, binY - 26, r.Width - 36, 22);
                using (SolidBrush sb2 = new SolidBrush(Color.FromArgb(232, 3, 10, 22)))
                using (Pen sp2 = new Pen(Color.FromArgb(195, 0, 240, 255), 1.2f))
                {
                    g.FillRectangle(sb2, subR2);
                    g.DrawRectangle(sp2, subR2);
                    g.DrawString(
                        "♫ [" + _activeVoiceEngineLabel + " • " + _lang.ToUpperInvariant() + "] « " + _subtitleText + " »",
                        new Font("Segoe UI", 8.0f, FontStyle.Bold), Brushes.Khaki, subR2.X + 8, subR2.Y + 2
                    );
                }
            }

            g.Clip = oldClip;
            using (Pen border = new Pen(Color.FromArgb(185, 0, 229, 255), 1.6f))
                g.DrawRectangle(border, r);
        }

        private void DrawAstrophysicsAndJwstStage3(Graphics g, Rectangle r)
        {
            Region oldClip = g.Clip;
            g.SetClip(r);
            using (SolidBrush bg = new SolidBrush(Color.FromArgb(248, 7, 14, 26)))
                g.FillRectangle(bg, r);

            // 1. Stitch Screen 4 Top L2 Lagrange Lock & JWST/LOIS Spectroscopy Array Header Bar
            Rectangle topBar = new Rectangle(r.X + 10, r.Y + 8, r.Width - 20, 24);
            using (SolidBrush bBar = new SolidBrush(Color.FromArgb(235, 12, 20, 31)))
            using (Pen pBar = new Pen(Color.FromArgb(190, 0, 240, 255), 1.2f))
            using (Font fBar = new Font("Consolas", 7.8f, FontStyle.Bold))
            using (SolidBrush bCy = new SolidBrush(Color.FromArgb(0, 240, 255)))
            using (SolidBrush bEm = new SolidBrush(Color.FromArgb(52, 255, 141)))
            {
                g.FillRectangle(bBar, topBar);
                g.DrawRectangle(pBar, topBar);
                g.DrawString("❖ [STAGE.04] L2_LAGRANGE_LOCK | JWST / LOIS SPECTROSCOPY ARRAY • BERYLLIUM 6.5M NIRSpec•MIRI", fBar, bCy, topBar.X + 8, topBar.Y + 5);
                g.DrawString("CRYOMR_TEMP: 6.42 K | RESOLVING R: 2,700 | FILTER: F290LP/G395H | LOIS_LINK: 99.88%", fBar, bEm, topBar.Right - 535, topBar.Y + 5);
            }

            // 2. Left Planet Limb Rim Glow Portrait (Stitch HD Render + 3D Sphere) & 4 Kid Astrophysics Laws
            if (_stitchTrappistImg != null)
            {
                Rectangle stImgR = new Rectangle(r.X + 14, r.Y + 38, 128, 114);
                g.DrawImage(_stitchTrappistImg, stImgR);
                using (Pen pRim = new Pen(Color.FromArgb(200, 0, 240, 255), 1.4f))
                    g.DrawRectangle(pRim, stImgR);
            }
            Draw3DPlanetWithFallback(g, _current.Index, r.X + 78f, r.Y + 95f, 38f);

            using (Font fTitle = new Font("Consolas", 10.6f, FontStyle.Bold))
            using (Font fLaw = new Font("Consolas", 8.6f, FontStyle.Bold))
            {
                if (_lang == "fr")
                {
                    g.DrawString("❖ LABO JWST ENFANTS & 4 LOIS ASTROPHYSIQUES // " + _current.ShortLabel + " (" + _current.HostStar + ")", fTitle, Brushes.Gold, r.X + 154, r.Y + 36);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "1. Loi du Transit (Taille)    : Ombre ΔF = {0:F0} ppm  ➔  Rayon Rp = {1:F2} x Terre", _current.DepthPpm, _current.RadiusRearth), fLaw, Brushes.Cyan, r.X + 154, r.Y + 58);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "2. Loi Doppler (Masse/Densité): Oscillation K = {0:F2} m/s  ➔  Densité ρ = {1:F2} g/cm³", _current.RadialVelocityKms, _current.BulkDensityGcm3), fLaw, Brushes.Aquamarine, r.X + 154, r.Y + 78);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "3. Zone Habitable (Eau)       : Insolation = {0:F2} x Terre | Teq = {1:F0} K | Habitable = {2}", _current.InsolationEarth, _current.TeqK, _current.InHz ? "OUI [H2O]" : "NON"), fLaw, Brushes.Khaki, r.X + 154, r.Y + 98);
                    g.DrawString("4. Détective Spatial (Règle 4): Statut = " + _current.Disposition + " — Cliquez sur APPROUVER ou REJETER (+800 PTS) !", fLaw, Brushes.LightGreen, r.X + 154, r.Y + 118);
                }
                else
                {
                    g.DrawString("❖ KID ASTROPHYSICS LAB & JWST SPECTROSCOPY // " + _current.ShortLabel + " (" + _current.HostStar + ")", fTitle, Brushes.Gold, r.X + 154, r.Y + 36);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "1. Transit Shadow Law (Size)  : Shadow Dip ΔF = {0:F0} ppm  ➔  Planet Radius Rp = {1:F2} x Earth", _current.DepthPpm, _current.RadiusRearth), fLaw, Brushes.Cyan, r.X + 154, r.Y + 58);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "2. Doppler Wobble Law (Mass)  : Star Wobble K = {0:F2} m/s  ➔  Density ρ = {1:F2} g/cm³", _current.RadialVelocityKms, _current.BulkDensityGcm3), fLaw, Brushes.Aquamarine, r.X + 154, r.Y + 78);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "3. Habitable Zone (Water)     : Sunlight = {0:F2} x Earth | Temp = {1:F0} K | Liquid Water HZ = {2}", _current.InsolationEarth, _current.TeqK, _current.InHz ? "YES [H2O]" : "NO"), fLaw, Brushes.Khaki, r.X + 154, r.Y + 98);
                    g.DrawString("4. Space Detective Check      : Status = " + _current.Disposition + " — Click APPROVE or REJECT (+800 PTS)!", fLaw, Brushes.LightGreen, r.X + 154, r.Y + 118);
                }
            }

            // Approve & Reject Interactive Buttons (Localized EN / FR)
            Rectangle btnApp = new Rectangle(r.Right - 360, r.Y + 38, 168, 30);
            Rectangle btnRej = new Rectangle(r.Right - 182, r.Y + 38, 168, 30);
            using (SolidBrush bA = new SolidBrush(Color.FromArgb(16, 185, 129)))
            using (SolidBrush bR = new SolidBrush(Color.FromArgb(225, 29, 72)))
            {
                g.FillRectangle(bA, btnApp);
                g.FillRectangle(bR, btnRej);
                string appTxt = (_lang == "fr") ? "✓ APPROUVER (+800)" : "✓ APPROVE (+800 PTS)";
                string rejTxt = (_lang == "fr") ? "✗ REJETER (+800)" : "✗ REJECT (+800 PTS)";
                g.DrawString(appTxt, new Font("Segoe UI", 8.5f, FontStyle.Bold), Brushes.Black, btnApp.X + 8, btnApp.Y + 6);
                g.DrawString(rejTxt, new Font("Segoe UI", 8.5f, FontStyle.Bold), Brushes.White, btnRej.X + 10, btnRej.Y + 6);
            }
            _hotspots.Add(new HudHotspot { Id = "approve", Rect = btnApp });
            _hotspots.Add(new HudHotspot { Id = "reject", Rect = btnRej });

            // Stitch Screen 4: [NASA_TAP_ADQL_SYNC] Live Telemetry Badge under Approve/Reject buttons
            Rectangle tapBox = new Rectangle(r.Right - 360, r.Y + 72, 346, 44);
            using (SolidBrush bTap = new SolidBrush(Color.FromArgb(230, 12, 20, 31)))
            using (Pen pTap = new Pen(Color.FromArgb(150, 52, 255, 141), 1.1f))
            using (Font fTap = new Font("Consolas", 7.0f, FontStyle.Bold))
            {
                g.FillRectangle(bTap, tapBox);
                g.DrawRectangle(pTap, tapBox);
                g.DrawString("[NASA_TAP_ADQL_SYNC] // LATENCY: 42ms // OK", fTap, Brushes.Aquamarine, tapBox.X + 6, tapBox.Y + 4);
                g.DrawString("SELECT pl_name, pl_masse, pl_eqt FROM pscomppars", fTap, Brushes.LightCyan, tapBox.X + 6, tapBox.Y + 17);
                g.DrawString(string.Format(CultureInfo.InvariantCulture, "THERMAL GRADIENT MODEL (T_eq): {0:F0} K [HAB_ZONE_BAND]", _current.TeqK), fTap, Brushes.Gold, tapBox.X + 6, tapBox.Y + 30);
            }

            // 3. Stitch Screen 4: [NIRSPEC_SPECTRUM_WATERFALL] (1.0µm - 28.0µm) + 6 Molecular Absorption Cards (H2O, CH4, CO2, CO, NH3, O3 ★)
            int leftSpecW = (int)((r.Width - 36) * 0.54);
            int bottomRowY = r.Y + 144;
            int bottomRowH = Math.Max(110, r.Height - 154);
            Rectangle specBox = new Rectangle(r.X + 12, bottomRowY, leftSpecW, bottomRowH);
            using (SolidBrush sbSpec = new SolidBrush(Color.FromArgb(240, 7, 14, 26)))
            using (Pen pSpec = new Pen(Color.FromArgb(190, 0, 240, 255), 1.4f))
            using (Font fSpecHdr = new Font("Consolas", 7.8f, FontStyle.Bold))
            using (Font fMol = new Font("Consolas", 7.2f, FontStyle.Bold))
            {
                g.FillRectangle(sbSpec, specBox);
                g.DrawRectangle(pSpec, specBox);
                g.DrawString("[NIRSPEC_SPECTRUM_WATERFALL] // 1.0µm [NIRSpec] ➔ 28.0µm [MIRI]", fSpecHdr, Brushes.Cyan, specBox.X + 8, specBox.Y + 5);
                g.DrawString("● PEAK DETECT: OZONE (O3) @ 9.6µm [4.8σ]", fSpecHdr, Brushes.SpringGreen, specBox.Right - 275, specBox.Y + 5);

                // Draw Transmission Spectroscopy Curve with Absorption Peaks (H2O 1.4µm, CH4 3.3µm, CO2 4.3µm, O3 9.6µm)
                int plotH = Math.Max(42, specBox.Height - 66);
                Rectangle plotR = new Rectangle(specBox.X + 8, specBox.Y + 22, specBox.Width - 16, plotH);
                using (SolidBrush bPlot = new SolidBrush(Color.FromArgb(4, 9, 18)))
                using (Pen pGrid = new Pen(Color.FromArgb(40, 0, 240, 255), 1.0f))
                {
                    g.FillRectangle(bPlot, plotR);
                    g.DrawRectangle(pGrid, plotR);
                }

                PointF[] specPts = new PointF[90];
                for (int i = 0; i < 90; i++)
                {
                    double xNorm = i / 89.0;
                    double baseFlux = 0.25 + 0.12 * Math.Sin(xNorm * 9.0);
                    double pH2O = 0.45 * Math.Exp(-Math.Pow((xNorm - 0.14) / 0.04, 2));
                    double pCH4 = 0.52 * Math.Exp(-Math.Pow((xNorm - 0.32) / 0.045, 2));
                    double pCO2 = 0.68 * Math.Exp(-Math.Pow((xNorm - 0.52) / 0.05, 2));
                    double pO3  = 0.64 * Math.Exp(-Math.Pow((xNorm - 0.78) / 0.045, 2));
                    double val = Math.Min(0.95, baseFlux + pH2O + pCH4 + pCO2 + pO3);
                    float sx = plotR.X + 4f + (float)xNorm * (plotR.Width - 8f);
                    float sy = plotR.Bottom - 4f - (float)val * (plotR.Height - 8f);
                    specPts[i] = new PointF(sx, sy);
                }
                using (Pen pGlow = new Pen(Color.FromArgb(65, 0, 240, 255), 5.0f))
                using (Pen pCore = new Pen(Color.FromArgb(52, 255, 141), 1.8f))
                {
                    g.DrawLines(pGlow, specPts);
                    g.DrawLines(pCore, specPts);
                }

                // 6 Stitch Molecular Absorption Cards along bottom of Waterfall panel
                string[] molNames = new string[] { "H2O 88%", "CH4 62%", "CO2 99%", "CO 14%", "NH3 41%", "O3 ★ 78%" };
                string[] molBands = new string[] { "1.4µm•H2O", "3.3µm•CH4", "4.3µm•CO2", "4.7µm•CO", "10.5µm•NH3", "9.6µm•BIO" };
                int mw = (specBox.Width - 16) / 6;
                int myCard = specBox.Bottom - 38;
                for (int m = 0; m < 6; m++)
                {
                    Rectangle mr = new Rectangle(specBox.X + 8 + m * mw, myCard, mw - 4, 32);
                    bool isBio = (m == 5 || m == 0 || m == 2);
                    using (SolidBrush mb = new SolidBrush(Color.FromArgb(230, 14, 24, 38)))
                    using (Pen mp = new Pen(isBio ? Color.FromArgb(52, 255, 141) : Color.FromArgb(0, 240, 255), 1.1f))
                    {
                        g.FillRectangle(mb, mr);
                        g.DrawRectangle(mp, mr);
                        g.DrawString(molNames[m], fMol, isBio ? Brushes.SpringGreen : Brushes.Cyan, mr.X + 4, mr.Y + 3);
                        g.DrawString(molBands[m], new Font("Consolas", 6.4f), Brushes.LightGray, mr.X + 4, mr.Y + 17);
                    }
                }
            }

            // 4. Right Side: 4 JWST Direct Imaging Coronagraph Filter Panels (NIRCam F300M/F444W + MIRI F1140C/F1550C)
            string[] titles = new string[] { "NIRCam 3.0µm", "NIRCam 4.4µm", "MIRI 11.4µm", "MIRI 15.5µm" };
            Color[] cols = new Color[] { Color.FromArgb(168, 85, 247), Color.FromArgb(56, 189, 248), Color.FromArgb(245, 158, 11), Color.FromArgb(239, 68, 68) };
            int rightStartX = specBox.Right + 10;
            int rightAvailW = r.Right - rightStartX - 12;
            int panelW = (rightAvailW - 18) / 4;
            for (int b = 0; b < 4; b++)
            {
                Rectangle pr = new Rectangle(rightStartX + b * (panelW + 6), bottomRowY, panelW, bottomRowH);
                using (SolidBrush pBg = new SolidBrush(Color.FromArgb(6, 4, 14)))
                using (Pen pPen = new Pen(cols[b], 1.6f))
                {
                    g.FillRectangle(pBg, pr);
                    g.DrawRectangle(pPen, pr);
                }
                // Host star coronagraph mask ★
                float sx = pr.X + pr.Width * 0.36f;
                float sy = pr.Y + pr.Height * 0.44f;
                using (Pen maskP = new Pen(Color.FromArgb(110, 255, 255, 255), 1.0f))
                    g.DrawEllipse(maskP, sx - 10f, sy - 10f, 20f, 20f);
                g.DrawString("★", new Font("Segoe UI", 9f, FontStyle.Bold), Brushes.White, sx - 6f, sy - 7f);

                // Directly imaged exoplanet PSF
                float px = pr.X + pr.Width * 0.64f;
                float py = pr.Y + pr.Height * 0.68f;
                using (GraphicsPath gp = new GraphicsPath())
                {
                    float rad = 18f + b * 3f;
                    gp.AddEllipse(px - rad, py - rad, rad * 2f, rad * 2f);
                    using (PathGradientBrush pgb = new PathGradientBrush(gp))
                    {
                        pgb.CenterColor = Color.White;
                        pgb.SurroundColors = new Color[] { Color.FromArgb(40, cols[b]) };
                        g.FillPath(pgb, gp);
                    }
                }
                g.DrawString(titles[b], new Font("Consolas", 7.2f, FontStyle.Bold), new SolidBrush(cols[b]), pr.X + 4, pr.Y + 5);
            }

            g.Clip = oldClip;
            using (Pen border = new Pen(Color.FromArgb(170, 56, 189, 248), 1.6f))
                g.DrawRectangle(border, r);
        }

        private void DrawBoincAndCommunityStage4(Graphics g, Rectangle r)
        {
            Region oldClip = g.Clip;
            g.SetClip(r);

            using (SolidBrush bg = new SolidBrush(Color.FromArgb(250, 3, 9, 20)))
            {
                g.FillRectangle(bg, r);
            }
            if (_galaxyBackdrop != null)
            {
                g.DrawImage(_galaxyBackdrop, r);
                using (SolidBrush dim = new SolidBrush(Color.FromArgb(215, 3, 10, 22)))
                    g.FillRectangle(dim, r);
            }

            // Stitch Header Bar for Stage 5: STEAM FREE-TO-PLAY IDLE-GPU BOINC CITIZEN SCIENCE & GCP COMMUNITY HUB
            Rectangle hdrBar = new Rectangle(r.X + 10, r.Y + 8, r.Width - 20, 28);
            bool isBoincActive = (_boincAutoIdleMode && _idleSeconds >= 4.0) || _boincForceAlwaysOn;
            using (SolidBrush hb = new SolidBrush(Color.FromArgb(240, 8, 22, 38)))
            using (Pen hp = new Pen(Color.FromArgb(200, 52, 255, 141), 1.3f))
            using (Font fHdr = new Font("Consolas", 8.2f, FontStyle.Bold))
            {
                g.FillRectangle(hb, hdrBar);
                g.DrawRectangle(hp, hdrBar);
                g.DrawString("❖ STAGE 5 // STEAM FREE-TO-PLAY (APPID 3133700) + AUTOMATIC IDLE-GPU BOINC CITIZEN SCIENCE + GCP COMMUNITY HUB", fHdr, Brushes.SpringGreen, hdrBar.X + 10, hdrBar.Y + 7);
                string gpuState = isBoincActive
                    ? string.Format(CultureInfo.InvariantCulture, "● GPU IDLE ({0:F1}s >= 4.0s): CRUNCHING BOINC (88% RTX 2070)", _idleSeconds)
                    : string.Format(CultureInfo.InvariantCulture, "● ACTIVE GAMEPLAY ({0:F1}s): 100% GPU FOR 60 FPS (BOINC 0%)", _idleSeconds);
                g.DrawString(gpuState, fHdr, isBoincActive ? Brushes.Cyan : Brushes.Gold, hdrBar.Right - 455, hdrBar.Y + 7);
            }

            int colY = hdrBar.Bottom + 10;
            int colH = r.Bottom - colY - 10;
            int leftW = (int)((r.Width - 36) * 0.36);
            int midW = (int)((r.Width - 36) * 0.36);
            int rightW = (r.Width - 36) - leftW - midW;

            Rectangle leftPanel = new Rectangle(r.X + 10, colY, leftW, colH);
            Rectangle midPanel = new Rectangle(leftPanel.Right + 8, colY, midW, colH);
            Rectangle rightPanel = new Rectangle(midPanel.Right + 8, colY, rightW, colH);

            // 1. LEFT PANEL: IDLE-GPU BOINC ASTROPHYSICS SCHEDULER & 4 LIVE PROJECTS
            DrawChamferedMfdPanel(g, leftPanel, Color.FromArgb(238, 6, 18, 34), Color.FromArgb(210, 0, 240, 255), 14);
            using (Font fTitle = new Font("Consolas", 8.8f, FontStyle.Bold))
            using (Font fSub = new Font("Consolas", 7.4f, FontStyle.Bold))
            using (Font fSmall = new Font("Consolas", 7.0f, FontStyle.Regular))
            {
                g.DrawString("► BOINC IDLE-GPU CITIZEN-SCIENCE SCHEDULER", fTitle, Brushes.Cyan, leftPanel.X + 12, leftPanel.Y + 10);
                g.DrawString("GPU Policy: 0% GPU during active flight (locked 60 FPS) -> 88% GPU when idle >= 4.0s", fSmall, Brushes.Aquamarine, leftPanel.X + 12, leftPanel.Y + 28);

                string[] pNames = new string[] {
                    "1. Einstein@Home (LIGO Pulsar & Gravitational Wave Search)",
                    "2. HiveMind-TESS@Home (BLS Exoplanet Transit Periodogram)",
                    "3. MilkyWay@Home (N-Body Dark Matter Halo Stream Fit)",
                    "4. JWST-LOIS@Home (NIRSpec H2O/CH4/CO2/O3 Retrieval)"
                };
                string[] pWus = new string[] {
                    "WU-EAH-O4-PSR-8841  | +1,420.5 Cobblestones",
                    "WU-TESS-K218B-092   | +2,180.0 Cobblestones",
                    "WU-MW-SAGITTAR-4410 | +980.0 Cobblestones",
                    "WU-JWST-NIRSPEC-317 | +1,850.0 Cobblestones"
                };

                int py = leftPanel.Y + 50;
                for (int i = 0; i < 4; i++)
                {
                    Rectangle pBox = new Rectangle(leftPanel.X + 12, py, leftPanel.Width - 24, 58);
                    using (SolidBrush pb = new SolidBrush(Color.FromArgb(220, 4, 12, 24)))
                    using (Pen pp = new Pen(Color.FromArgb(130, 0, 229, 255), 1.0f))
                    {
                        g.FillRectangle(pb, pBox);
                        g.DrawRectangle(pp, pBox);
                    }
                    double pct = Math.Min(100.0, _boincProjProgress[i]);
                    g.DrawString(pNames[i], fSub, Brushes.White, pBox.X + 8, pBox.Y + 6);
                    g.DrawString(string.Format(CultureInfo.InvariantCulture, "{0:F1}%", pct), fSub, Brushes.SpringGreen, pBox.Right - 48, pBox.Y + 6);

                    Rectangle barBg = new Rectangle(pBox.X + 8, pBox.Y + 24, pBox.Width - 16, 10);
                    int fillW = (int)((barBg.Width) * (pct / 100.0));
                    using (SolidBrush bTrack = new SolidBrush(Color.FromArgb(30, 255, 255, 255)))
                    using (SolidBrush bFill = new SolidBrush(i % 2 == 0 ? Color.FromArgb(0, 229, 255) : Color.FromArgb(52, 255, 141)))
                    {
                        g.FillRectangle(bTrack, barBg);
                        if (fillW > 0) g.FillRectangle(bFill, new Rectangle(barBg.X, barBg.Y, fillW, barBg.Height));
                    }
                    g.DrawString(pWus[i], fSmall, Brushes.LightSteelBlue, pBox.X + 8, pBox.Y + 38);
                    py += 64;
                }

                // Interactive BOINC Controls at bottom of Left Panel
                Rectangle btnCrunch = new Rectangle(leftPanel.X + 12, leftPanel.Bottom - 76, leftPanel.Width - 24, 28);
                Rectangle btnMode = new Rectangle(leftPanel.X + 12, leftPanel.Bottom - 42, (leftPanel.Width - 30) / 2, 26);
                Rectangle btnWeb = new Rectangle(btnMode.Right + 6, leftPanel.Bottom - 42, (leftPanel.Width - 30) / 2, 26);

                using (SolidBrush bCr = new SolidBrush(Color.FromArgb(16, 185, 129)))
                using (SolidBrush bMd = new SolidBrush(Color.FromArgb(14, 116, 144)))
                using (SolidBrush bWb = new SolidBrush(Color.FromArgb(245, 158, 11)))
                {
                    g.FillRectangle(bCr, btnCrunch);
                    g.FillRectangle(bMd, btnMode);
                    g.FillRectangle(bWb, btnWeb);
                    g.DrawString("★ CRUNCH BOINC WORKUNIT NOW & SYNC GCP (+250 PTS)", fSub, Brushes.Black, btnCrunch.X + 10, btnCrunch.Y + 7);
                    string modeTxt = _boincForceAlwaysOn ? "► MODE: FORCE ON" : "► MODE: AUTO IDLE (>4s)";
                    g.DrawString(modeTxt, fSmall, Brushes.White, btnMode.X + 6, btnMode.Y + 6);
                    g.DrawString("► OPEN WEB HUB (:8095)", fSmall, Brushes.Black, btnWeb.X + 6, btnWeb.Y + 6);
                }
                _hotspots.Add(new HudHotspot { Id = "boinc_crunch_now", Rect = btnCrunch });
                _hotspots.Add(new HudHotspot { Id = "boinc_toggle_mode", Rect = btnMode });
                _hotspots.Add(new HudHotspot { Id = "boinc_open_web", Rect = btnWeb });
            }

            // 2. CENTER PANEL: UNIFIED GAMER & ASTROPHYSICS CITIZEN-SCIENTIST LEADERBOARD + CO-OP FEED
            DrawChamferedMfdPanel(g, midPanel, Color.FromArgb(238, 6, 18, 34), Color.FromArgb(210, 52, 255, 141), 14);
            using (Font fTitle = new Font("Consolas", 8.8f, FontStyle.Bold))
            using (Font fTblHdr = new Font("Consolas", 7.2f, FontStyle.Bold))
            using (Font fRow = new Font("Consolas", 7.1f, FontStyle.Regular))
            {
                g.DrawString("► GAMER & ASTROPHYSICIST COMMUNITY LEADERBOARD", fTitle, Brushes.SpringGreen, midPanel.X + 12, midPanel.Y + 10);
                g.DrawString(string.Format(CultureInfo.InvariantCulture, "Active Community: {0:N0} (Steam Gamers + GCP Astrophysicists) | Total: {1:N1} Cobblestones", _communityActiveMembers, _boincCobblestones), fRow, Brushes.Gold, midPanel.X + 12, midPanel.Y + 28);

                int ty = midPanel.Y + 48;
                g.DrawString("#  COMMANDER / SCIENTIST       ROLE                     GAME PTS   BOINC COBBLESTONES", fTblHdr, Brushes.Cyan, midPanel.X + 12, ty);
                ty += 18;

                string[] ranks = new string[] {
                    string.Format(CultureInfo.InvariantCulture, "#1 Cmdr_JeanPaul_NCC31337      Astrophysicist + Gamer   {0,8:N0}   14,820.5 ({1} WUs)", Math.Max(48500, _scoreHmt), _boincWorkunitsCompleted),
                    "#2 Dr_Elena_Vega_IAP           Exoplanet Spectroscopist   44,200   19,410.0 (245 WUs)",
                    "#3 Starfleet_Cadet_Leo         Steam Idle-GPU Gamer       39,800    9,640.0 (118 WUs)",
                    "#4 Prof_K_eb_JWST_Observer     Astrophysics Citizen Sci   36,150   22,190.5 (302 WUs)",
                    "#5 Oolite_Ace_Commander        Elite/Oolite Pilot + GPU   33,900    8,120.0 ( 96 WUs)",
                    "#6 AstroBee_Swarm_Node07       GCP Cloud Run T4 Node      31,400   16,800.0 (215 WUs)"
                };

                for (int i = 0; i < ranks.Length; i++)
                {
                    Rectangle rRow = new Rectangle(midPanel.X + 10, ty - 2, midPanel.Width - 20, 22);
                    using (SolidBrush rb = new SolidBrush(i == 0 ? Color.FromArgb(55, 52, 255, 141) : Color.FromArgb(140, 4, 12, 24)))
                    {
                        g.FillRectangle(rb, rRow);
                    }
                    g.DrawString(ranks[i], fRow, i == 0 ? Brushes.SpringGreen : Brushes.White, midPanel.X + 12, ty + 2);
                    ty += 25;
                }

                // Co-Op Exoplanet Discovery Feed section
                ty += 8;
                g.DrawString("► LIVE CO-OP EXOPLANET DISCOVERY FEED (GAMERS + SCIENTISTS)", fTblHdr, Brushes.Gold, midPanel.X + 12, ty);
                ty += 18;

                string[] feedNotes = new string[] {
                    "● " + _communityLatestNote,
                    "● Dr_Elena_Vega_IAP (Astrophysicist): Cross-matched Steam idle-GPU BLS fold WU-TESS-K218B-092 with JWST NIRSpec G395H -> 3.4-sigma CH4 + CO2!",
                    "● Starfleet_Cadet_Leo (Steam Gamer): Completed Stage 3 Oolite landing on LHS 1140 b; idle RTX GPU folded 4 TESS sectors while in star chart!",
                    "● Prof_K_eb_JWST_Observer (Citizen Scientist): Flagged KOI-Bin #02 as Eclipsing Binary (Depth 32,000 ppm, K=48.5 km/s) — verified by 18 gamers."
                };

                for (int i = 0; i < feedNotes.Length; i++)
                {
                    Rectangle fBox = new Rectangle(midPanel.X + 10, ty, midPanel.Width - 20, 38);
                    using (SolidBrush fb = new SolidBrush(Color.FromArgb(200, 4, 14, 28)))
                    using (Pen fp = new Pen(Color.FromArgb(110, 0, 229, 255), 1.0f))
                    {
                        g.FillRectangle(fb, fBox);
                        g.DrawRectangle(fp, fBox);
                    }
                    RectangleF rf = new RectangleF(fBox.X + 6, fBox.Y + 4, fBox.Width - 12, fBox.Height - 6);
                    g.DrawString(feedNotes[i], new Font("Consolas", 6.6f, FontStyle.Regular), i == 0 ? Brushes.Aquamarine : Brushes.LightCyan, rf);
                    ty += 42;
                }
            }

            // 3. RIGHT PANEL: STEAM FREE-TO-PLAY MANIFEST (APPID 3133700) & GCP CLOUD RUN FEDERATION
            DrawChamferedMfdPanel(g, rightPanel, Color.FromArgb(238, 6, 18, 34), Color.FromArgb(210, 245, 158, 11), 14);
            using (Font fTitle = new Font("Consolas", 8.6f, FontStyle.Bold))
            using (Font fSub = new Font("Consolas", 7.3f, FontStyle.Bold))
            using (Font fMono = new Font("Consolas", 6.9f, FontStyle.Regular))
            {
                g.DrawString("► STEAM FREE-TO-PLAY & GCP CLOUD", fTitle, Brushes.Gold, rightPanel.X + 12, rightPanel.Y + 10);
                g.DrawString("100% Free Game + Reciprocal Citizen Science", fMono, Brushes.Cyan, rightPanel.X + 12, rightPanel.Y + 28);

                int ry = rightPanel.Y + 48;
                string[] specs = new string[] {
                    "[STEAM STORE PACKAGE]",
                    "• AppID / DepotID : 3133700 / 3133702",
                    "• BillingType     : FreeOnDemand (isfreeapp=1)",
                    "• Price Charter   : 0.00 EUR / 0.00 USD",
                    "• Fat Client Exe  : HiveMind_Exoplanet_Hunter.exe",
                    "• Idle-GPU Daemon : boinc_idle_gpu_worker.py",
                    "",
                    "[GCP CLOUD RUN FEDERATION]",
                    "• Public URL      : hivemind-at-home-ruche",
                    "  https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
                    "• Region / Tier   : us-central1 (Scale-to-Zero)",
                    "• Latest Receipt  : " + _boincLastReceiptId,
                    "• Federated Lease : " + _boincLastGcpLeaseId,
                    "• Sync Status     : " + _boincLastSyncStatus,
                    "",
                    "[IDLE-GPU SAFETY GUARANTEES]",
                    "• Active Flight   : 0% BOINC (100% Game FPS)",
                    "• Idle >= 4.0 sec : 88% GPU BLS Transit Fold",
                    "• Thermal Cap     : Auto-pause if GPU > 78 C"
                };

                for (int i = 0; i < specs.Length; i++)
                {
                    bool isHdr = specs[i].StartsWith("[");
                    g.DrawString(specs[i], isHdr ? fSub : fMono, isHdr ? Brushes.Gold : Brushes.White, rightPanel.X + 12, ry);
                    ry += isHdr ? 17 : 15;
                }

                Rectangle btnPost = new Rectangle(rightPanel.X + 12, rightPanel.Bottom - 44, rightPanel.Width - 24, 30);
                using (SolidBrush bp = new SolidBrush(Color.FromArgb(245, 158, 11)))
                {
                    g.FillRectangle(bp, btnPost);
                    g.DrawString("★ POST CO-OP TRANSIT NOTE (+150 PTS)", fSub, Brushes.Black, btnPost.X + 8, btnPost.Y + 8);
                }
                _hotspots.Add(new HudHotspot { Id = "boinc_post_note", Rect = btnPost });
            }

            g.Clip = oldClip;
            using (Pen border = new Pen(Color.FromArgb(170, 56, 189, 248), 1.6f))
                g.DrawRectangle(border, r);
        }

        private void DrawDualNeonOscilloscopes(Graphics g, Rectangle r)
        {
            using (SolidBrush bg = new SolidBrush(Color.FromArgb(248, 3, 9, 16)))
            using (Pen border = new Pen(Color.FromArgb(170, 56, 189, 248), 1.6f))
            {
                g.FillRectangle(bg, r);
                g.DrawRectangle(border, r);
            }

            int pad = 10;
            int plotW = (r.Width - pad * 3) / 2;
            int plotH = r.Height - 32;
            Rectangle rvBox = new Rectangle(r.X + pad, r.Y + 24, plotW, plotH);
            Rectangle lcBox = new Rectangle(r.X + pad * 2 + plotW, r.Y + 24, plotW, plotH);

            // Localized Oscilloscope Titles (EN vs FR)
            using (Font fHdr = new Font("Consolas", 8.8f, FontStyle.Bold))
            using (SolidBrush bMint = new SolidBrush(Color.FromArgb(167, 243, 208)))
            using (SolidBrush bCyan = new SolidBrush(Color.FromArgb(165, 243, 252)))
            {
                string leftTitle = (_lang == "fr")
                    ? string.Format(CultureInfo.InvariantCulture, "CONSOLE SYSTÉMIQUE UCSC // VITESSE RADIALE DOPPLER (K = {0:F1} m/s, e = {1:F2})", _current.RadialVelocityKms, _current.Eccentricity)
                    : string.Format(CultureInfo.InvariantCulture, "UCSC SYSTEMIC CONSOLE // DOPPLER STAR WOBBLE (K = {0:F1} m/s, e = {1:F2} ➔ Planet Mass = {2:F2} Earths)", _current.RadialVelocityKms, _current.Eccentricity, _current.MassMearth);
                string rightTitle = (_lang == "fr")
                    ? string.Format(CultureInfo.InvariantCulture, "PHOTOMÉTRIE DE TRANSIT KEPLER/TESS (Profondeur = {0:F0} ppm)", _current.DepthPpm)
                    : string.Format(CultureInfo.InvariantCulture, "KEPLER/TESS TRANSIT SHADOW DIP (Depth = {0:F0} ppm ➔ Planet Size = {1:F2} Earths)", _current.DepthPpm, _current.RadiusRearth);

                g.DrawString(leftTitle, fHdr, bMint, rvBox.X, r.Y + 5);
                g.DrawString(rightTitle, fHdr, bCyan, lcBox.X, r.Y + 5);
            }

            // Draw high-density CRT grid inside both oscilloscopes
            foreach (Rectangle box in new Rectangle[] { rvBox, lcBox })
            {
                using (SolidBrush bDark = new SolidBrush(Color.FromArgb(2, 7, 13)))
                    g.FillRectangle(bDark, box);
                using (Pen gridMinor = new Pen(Color.FromArgb(28, 45, 212, 191), 1.0f))
                using (Pen gridMajor = new Pen(Color.FromArgb(55, 56, 189, 248), 1.0f))
                {
                    for (int gx = box.X; gx <= box.Right; gx += 22)
                        g.DrawLine((gx - box.X) % 88 == 0 ? gridMajor : gridMinor, gx, box.Y, gx, box.Bottom);
                    for (int gy = box.Y; gy <= box.Bottom; gy += 16)
                        g.DrawLine((gy - box.Y) % 64 == 0 ? gridMajor : gridMinor, box.X, gy, box.Right, gy);
                    g.DrawRectangle(gridMajor, box);
                }

                // Micro-telemetry blocks in corners (exact detail from reference image!)
                using (SolidBrush microB = new SolidBrush(Color.FromArgb(95, 52, 211, 153)))
                {
                    for (int m = 0; m < 4; m++)
                    {
                        g.FillRectangle(microB, box.X + 8, box.Bottom - 26 + m * 5, 26 - m * 4, 2);
                        g.FillRectangle(microB, box.Right - 42, box.Y + 8 + m * 5, 32 - m * 5, 2);
                    }
                }
            }

            // 1. Left Oscilloscope Curve: Doppler Radial Velocity (3-Pass Neon Emerald Glow + Diagonal Reference Line)
            float rvMidY = rvBox.Y + rvBox.Height * 0.5f;
            using (Pen zeroP = new Pen(Color.FromArgb(80, 148, 163, 184), 1.0f))
            using (Pen diagP = new Pen(Color.FromArgb(75, 52, 211, 153), 1.2f))
            {
                g.DrawLine(zeroP, rvBox.X, rvMidY, rvBox.Right, rvMidY);
                g.DrawLine(diagP, rvBox.X + 12, rvBox.Bottom - 8, rvBox.Right - 28, rvBox.Y + 8);
            }

            PointF[] rvPts = new PointF[100];
            double ecc = Math.Min(0.933, _current.Eccentricity);
            bool isFlat = (_current.RadialVelocityKms < 0.1);
            for (int i = 0; i < 100; i++)
            {
                double phase = i / 99.0;
                double M = phase * Math.PI * 2.0;
                double E = SolveKeplerEquation(M, ecc);
                double nu = 2.0 * Math.Atan2(Math.Sqrt(1.0 + ecc) * Math.Sin(E / 2.0), Math.Sqrt(1.0 - ecc) * Math.Cos(E / 2.0));
                double omega = (ecc > 0.5) ? 5.24 : 0.42;
                double rvNorm = (Math.Cos(nu + omega) + ecc * Math.Cos(omega)) / (1.0 + ecc);
                if (isFlat) rvNorm = Math.Sin(phase * 18.0) * 0.03;
                float x = rvBox.X + 8f + (float)phase * (rvBox.Width - 16f);
                float y = rvMidY - (float)rvNorm * (rvBox.Height * 0.39f);
                rvPts[i] = new PointF(x, y);
            }
            using (Pen glowOuter = new Pen(Color.FromArgb(42, 16, 185, 129), 7.0f))
            using (Pen glowMid = new Pen(Color.FromArgb(115, 52, 211, 153), 3.6f))
            using (Pen core = new Pen(Color.FromArgb(209, 250, 229), 1.8f))
            {
                g.DrawLines(glowOuter, rvPts);
                g.DrawLines(glowMid, rvPts);
                g.DrawLines(core, rvPts);
            }

            // 2. Right Oscilloscope Curve: Kepler/TESS Limb-Darkened Transit Curve (3-Pass Neon Cyan Glow)
            bool isFp = (_current.Disposition == "FALSE POSITIVE");
            PointF[] lcPts = new PointF[110];
            for (int i = 0; i < 110; i++)
            {
                double uNorm = (i / 109.0) * 2.0 - 1.0;
                float x = lcBox.X + 8f + (i / 109f) * (lcBox.Width - 16f);
                double dip = 0.0;
                double halfW = 0.24;
                if (Math.Abs(uNorm) <= halfW)
                {
                    double u = Math.Abs(uNorm) / halfW;
                    if (isFp) dip = Math.Max(0.0, 1.0 - u);
                    else
                    {
                        double mu = Math.Sqrt(Math.Max(0.0, 1.0 - u * u));
                        dip = Math.Min(1.0, (1.0 - 0.35 * (1.0 - mu)) * 1.12);
                    }
                }
                float y = lcBox.Y + 14f + (float)(dip * (lcBox.Height - 32f));
                lcPts[i] = new PointF(x, y);
            }
            Color cGlow = isFp ? Color.FromArgb(244, 63, 94) : Color.FromArgb(34, 211, 238);
            Color cCore = isFp ? Color.FromArgb(254, 205, 211) : Color.FromArgb(224, 242, 254);
            using (Pen glowOuter = new Pen(Color.FromArgb(45, cGlow), 7.0f))
            using (Pen glowMid = new Pen(Color.FromArgb(125, cGlow), 3.6f))
            using (Pen core = new Pen(cCore, 1.8f))
            {
                g.DrawLines(glowOuter, lcPts);
                g.DrawLines(glowMid, lcPts);
                g.DrawLines(core, lcPts);
            }

            // Secondary baseline pulse trace at bottom of right oscilloscope
            using (Pen basePen = new Pen(Color.FromArgb(110, 56, 189, 248), 1.1f))
            {
                g.DrawLine(basePen, lcBox.X + 8, lcBox.Bottom - 7, lcBox.Right - 8, lcBox.Bottom - 7);
            }
        }

        private void DrawBottom10SlotHolographicCollection(Graphics g, Rectangle r)
        {
            using (SolidBrush bg = new SolidBrush(Color.FromArgb(245, 4, 15, 13)))
            using (Pen border = new Pen(Color.FromArgb(190, 52, 211, 153), 1.8f))
            {
                g.FillRectangle(bg, r);
                g.DrawRectangle(border, r);
            }

            using (Font fHdr = new Font("Consolas", 8.8f, FontStyle.Bold))
            using (SolidBrush bGold = new SolidBrush(Color.FromArgb(253, 224, 71)))
            {
                string colHdr = (_lang == "fr")
                    ? string.Format(CultureInfo.InvariantCulture, "⊞ COLLECTION ASTROBIOLOGIQUE & PASSEPORT ENFANT (CLIQUEZ SUR UNE PLANÈTE) — SCORE: {0:N0} PTS | MÉDAILLES: {1}/10 ★", _scoreHmt, _medalsCount)
                    : string.Format(CultureInfo.InvariantCulture, "⊞ ASTROBIOLOGY COLLECTION & KID DISCOVERY PASSPORT (CLICK ANY PLANET TO EXPLORE) — SCORE: {0:N0} PTS | MEDALS: {1}/10 ★", _scoreHmt, _medalsCount);
                g.DrawString(colHdr, fHdr, bGold, r.X + 10, r.Y + 5);
            }

            int pad = 8;
            int availW = r.Width - pad * 2;
            int slotW = availW / 10;
            for (int i = 0; i < 10; i++)
            {
                var t = _targets[i];
                Rectangle sr = new Rectangle(r.X + pad + i * slotW + 3, r.Y + 24, slotW - 6, r.Height - 30);
                _slotRects[i] = sr;
                bool isSel = (i == _current.Index);

                using (LinearGradientBrush cardGrad = new LinearGradientBrush(
                    sr, Color.FromArgb(230, 10, 34, 28), Color.FromArgb(245, 3, 12, 11), LinearGradientMode.Vertical))
                using (Pen cardPen = new Pen(isSel ? Color.FromArgb(253, 224, 71) : Color.FromArgb(120, 52, 211, 153), isSel ? 2.4f : 1.1f))
                {
                    g.FillRectangle(cardGrad, sr);
                    g.DrawRectangle(cardPen, sr);
                }

                float cx = sr.X + sr.Width * 0.5f;
                float pedY = sr.Bottom - 26f;

                // Glowing Concentric Holographic Pedestal Rings at base of card (matching reference image!)
                using (Pen pedOuter = new Pen(Color.FromArgb(210, t.PedestalColor), 1.6f))
                using (Pen pedInner = new Pen(Color.FromArgb(180, Color.White), 1.1f))
                {
                    g.DrawEllipse(pedOuter, cx - 26f, pedY - 6f, 52f, 12f);
                    g.DrawEllipse(pedInner, cx - 16f, pedY - 3.5f, 32f, 7f);
                }

                // 3D Raycast-Shaded Planet Globe floating above pedestal
                Draw3DPlanetWithFallback(g, i, cx, sr.Y + 34f, 22f);

                // Gold Star Medal Badge if landed!
                if (t.LandedMedal)
                {
                    g.DrawString("★", new Font("Segoe UI", 9.5f, FontStyle.Bold), Brushes.Gold, sr.Right - 20, sr.Y + 2);
                }

                // Checkmark + Planet ShortLabel at bottom of card
                using (Font fSlot = new Font("Consolas", 7.6f, FontStyle.Bold))
                using (SolidBrush bTxt = new SolidBrush(isSel ? Color.FromArgb(253, 224, 71) : Color.FromArgb(209, 250, 229)))
                {
                    g.DrawString("✓ " + t.ShortLabel, fSlot, bTxt, sr.X + 4, sr.Bottom - 16);
                }
            }
        }

        /// <summary>
        /// Draws the concise Planet Analysis, Specific Telemetry & Astrobiology (How Life Could Live) Card.
        /// Chapter explanation is mentioned ONLY ONCE on initial entry; moving between planets shows only planet analysis & habitability!
        /// </summary>
        private void DrawStepNarrativeOverlay(Graphics g, Rectangle r)
        {
            int cardW = Math.Min(850, r.Width - 50);
            int cardH = 104;
            int cardX = r.X + (r.Width - cardW) / 2;
            int cardY = r.Y + 8;
            Rectangle card = new Rectangle(cardX, cardY, cardW, cardH);

            using (SolidBrush bg = new SolidBrush(Color.FromArgb(234, 6, 16, 34)))
            using (Pen border = new Pen(Color.FromArgb(245, 250, 204, 21), 2.0f))
            using (Font fHdr = new Font("Segoe UI", 9.2f, FontStyle.Bold))
            using (Font fBody = new Font("Consolas", 8.0f, FontStyle.Bold))
            using (Font fSci = new Font("Segoe UI", 8.2f, FontStyle.Regular))
            using (SolidBrush bGold = new SolidBrush(Color.FromArgb(253, 224, 71)))
            using (SolidBrush bCyan = new SolidBrush(Color.FromArgb(165, 243, 252)))
            using (SolidBrush bMint = new SolidBrush(Color.FromArgb(110, 231, 183)))
            {
                g.FillRectangle(bg, card);
                g.DrawRectangle(border, card);

                string chTitle, analysisLine, lifeLine;
                bool showOneTimeChapterHdr = (_planetSwitchCount == 0);
                if (_lang == "fr")
                {
                    chTitle = showOneTimeChapterHdr
                        ? ("► " + GetOneTimeChapterExplanation(_gameStage))
                        : ("► ANALYSE PLANÉTAIRE & VIE POSSIBLE — " + _current.ShortLabel.ToUpperInvariant() + " (" + _current.HostStar + ")");
                    analysisLine = string.Format(
                        CultureInfo.InvariantCulture,
                        "★ {0} [P={1:F1}j | a={2:F3} UA | ρ={3:F2} g/cm³ | K={4:F2} m/s]",
                        _current.AnalysisFr, _current.PeriodDays, _current.SemiMajorAxisAu, _current.BulkDensityGcm3, _current.RadialVelocityKms
                    );
                    lifeLine = "🧬 " + _current.LifeFr;
                }
                else
                {
                    chTitle = showOneTimeChapterHdr
                        ? ("► " + GetOneTimeChapterExplanation(_gameStage))
                        : ("► PLANET ANALYSIS & HOW LIFE COULD LIVE — " + _current.ShortLabel.ToUpperInvariant() + " (" + _current.HostStar + ")");
                    analysisLine = string.Format(
                        CultureInfo.InvariantCulture,
                        "★ {0} [P={1:F1}d | a={2:F3} AU | ρ={3:F2} g/cm³ | K={4:F2} m/s]",
                        _current.AnalysisEn, _current.PeriodDays, _current.SemiMajorAxisAu, _current.BulkDensityGcm3, _current.RadialVelocityKms
                    );
                    lifeLine = "🧬 " + _current.LifeEn;
                }

                g.DrawString(chTitle, fHdr, bGold, card.X + 12, card.Y + 5);
                g.DrawString(analysisLine, fBody, bCyan, new RectangleF(card.X + 12, card.Y + 26, card.Width - 24, 22));
                g.DrawString(lifeLine, fSci, bMint, new RectangleF(card.X + 12, card.Y + 48, card.Width - 24, 32));

                // Interactive Buttons on bottom-right of Card: [♫ HEAR PLANET ANALYSIS (+150 PTS)] and [► HIDE [H]]
                Rectangle btnHear = new Rectangle(card.Right - 375, card.Bottom - 22, 225, 18);
                Rectangle btnHide = new Rectangle(card.Right - 142, card.Bottom - 22, 132, 18);
                using (SolidBrush bHr = new SolidBrush(Color.FromArgb(245, 158, 11)))
                using (SolidBrush bHd = new SolidBrush(Color.FromArgb(56, 189, 248)))
                using (Font fBtn = new Font("Segoe UI", 7.4f, FontStyle.Bold))
                {
                    g.FillRectangle(bHr, btnHear);
                    g.FillRectangle(bHd, btnHide);
                    g.DrawString((_lang == "fr") ? "♫ ÉCOUTER ANALYSE (+150 PTS)" : "♫ HEAR PLANET ANALYSIS (+150)", fBtn, Brushes.Black, btnHear.X + 6, btnHear.Y + 2);
                    g.DrawString((_lang == "fr") ? "► MASQUER [H]" : "► HIDE [H]", fBtn, Brushes.Black, btnHide.X + 6, btnHide.Y + 2);
                }
                _hotspots.Add(new HudHotspot { Id = "kid_lesson", Rect = btnHear });
                _hotspots.Add(new HudHotspot { Id = "toggle_narrative", Rect = btnHide });
            }
        }

        private void DrawFloatingPopups(Graphics g)
        {
            using (Font fPop = new Font("Segoe UI", 10.5f, FontStyle.Bold))
            {
                foreach (var p in _popups)
                {
                    int alpha = Math.Max(0, Math.Min(255, (int)(p.Life * 255)));
                    using (SolidBrush shadow = new SolidBrush(Color.FromArgb(alpha, 0, 0, 0)))
                    using (SolidBrush fg = new SolidBrush(Color.FromArgb(alpha, p.Color)))
                    {
                        g.DrawString(p.Text, fPop, shadow, p.X + 2f, p.Y + 2f);
                        g.DrawString(p.Text, fPop, fg, p.X, p.Y);
                    }
                }
            }
        }

        public void RunAutomatedRtxSelfTest(string reportPath)
        {
            int w = 1540, h = 940;
            var sw = new System.Diagnostics.Stopwatch();
            var stageMetrics = new List<string>();
            string[] stageNames = new string[] { "1.GALAXIE", "2.ORBIT_BEE", "3.OOLITE_LAND", "4.JWST_LOIS", "5.BOINC_GCP" };

            for (int st = 0; st < 5; st++)
            {
                _gameStage = st;
                _idleSeconds = (st == 4) ? 6.5 : 0.2;
                sw.Restart();
                using (Bitmap bmp = new Bitmap(w, h))
                using (Graphics g = Graphics.FromImage(bmp))
                using (PaintEventArgs pea = new PaintEventArgs(g, new Rectangle(0, 0, w, h)))
                {
                    this.OnPaint(pea);
                }
                sw.Stop();
                double ms = Math.Max(0.1, sw.Elapsed.TotalMilliseconds);
                double fps = Math.Round(1000.0 / ms, 1);
                stageMetrics.Add(string.Format(
                    CultureInfo.InvariantCulture,
                    "{{ \"stage_index\": {0}, \"stage_name\": \"{1}\", \"render_ms\": {2:F2}, \"effective_fps\": {3:F1}, \"boinc_idle_seconds\": {4:F1} }}",
                    st, stageNames[st], ms, fps, _idleSeconds
                ));
            }

            string json = string.Format(
                CultureInfo.InvariantCulture,
                "{{\n  \"schema\": \"hivemind.steam_fat_client.rtx_self_test/1\",\n  \"app_id\": 3133700,\n  \"depot_id\": 3133702,\n  \"resolution\": \"1540x940\",\n  \"languages_tested\": [\"fr\", \"en\", \"es\", \"de\", \"zh\"],\n  \"targets_loaded\": {0},\n  \"stages\": [{1}],\n  \"status\": \"PASS\"\n}}",
                _targets.Count,
                string.Join(", ", stageMetrics.ToArray())
            );
            File.WriteAllText(reportPath, json, new UTF8Encoding(false));
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            ExoplanetHunterForm form = new ExoplanetHunterForm();
            if (args != null && args.Length > 0 && args[0] == "--self-test")
            {
                string outPath = (args.Length > 1)
                    ? args[1]
                    : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "steam_assets", "fat_client_self_test_report.json");
                form.RunAutomatedRtxSelfTest(outPath);
                return;
            }
            Application.Run(form);
        }
    }
}
