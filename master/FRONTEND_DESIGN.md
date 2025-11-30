# 🎨 FightIQ Frontend: The "Octagon Terminal"
**Vision:** A premium, high-performance dashboard that visualizes the "Value Sniper" engine. Think "Bloomberg Terminal meets UFC".

## 1. Design Philosophy
*   **Aesthetic:** "Dark Mode Cyberpunk". Deep blacks, neon greens for profit/edge, stark whites for text.
*   **UX:** Density. Investors want data. No wasted whitespace.
*   **Vibe:** Professional, Institutional, Algorithmic.

## 2. Tech Stack
*   **Framework:** Next.js (React) - Fast, SEO-friendly.
*   **Styling:** Tailwind CSS + Shadcn/UI (Premium components).
*   **Charts:** Recharts (for Elo curves and Probability distributions).
*   **Icons:** Lucide React.
*   **State:** Zustand (Lightweight state management).

## 3. Core Views (Pages)

### A. The Command Center (Home)
*   **The "Sniper Feed":** A scrolling ticker of active bets with Edge > 5%.
    *   *Visual:* A "Lock" icon 🔒 next to high-confidence picks (e.g., Taira).
*   **Bankroll Pulse:** A live graph of our theoretical portfolio performance (+610% line).
*   **Upcoming Card:** A grid of the next event (UFC 323) with "Buy/Pass" indicators.

### B. The Matchup Lab (Fight Detail)
*   **Holographic Tale of the Tape:**
    *   Elo vs Elo (Dynamic Chart).
    *   Chin Health Bar (Decaying bar for older fighters).
    *   Reach/Age Differential (Visual bars).
*   **The Probability Gauge:** A donut chart showing Model Prob vs Implied Prob. The "Gap" is the Edge.
*   **Prop Hunter Matrix:** A heatmap of Method/Round probabilities.

### C. The Market Scanner (Live Odds)
*   **Steam Tracker:** Real-time line movement.
    *   *Visual:* Green arrows for "Sharp Money" coming in on our picks.
*   **Arb Finder:** (Future) Comparing DraftKings vs FanDuel lines.

## 4. Component Library (The "FightUI")
*   **`EdgeBadge`:** A pill component that glows Green if Edge > 5%, Yellow if > 2%, Gray if No Value.
*   **`ConfidenceMeter`:** A radial gauge from 0-100%.
*   **`FighterAvatar`:** Circular image with a colored border indicating "Form" (Hot/Cold).

## 5. Implementation Roadmap
1.  **Phase 1:** Setup Next.js + Tailwind. Build the "Command Center" with static data (UFC 323 JSON).
2.  **Phase 2:** Build the "Matchup Lab" component.
3.  **Phase 3:** Connect to the Python Backend (FastAPI) for live inference.
