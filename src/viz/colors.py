from pygame import Color

# Node states
NODE_UNVISITED: Color = Color(230, 90, 70)  # soft red — unassigned
NODE_ASSIGNED: Color = Color(70, 150, 130)  # teal — planned
NODE_DEPOT: Color = Color(70, 90, 210)  # blue — depot
NODE_VISITED: Color = Color(190, 190, 190)  # light gray — visited

# Trucks
TRUCK_BROKEN: Color = Color(170, 170, 170)  # mid gray

# Route lines — slightly darker for visibility on white
ROUTE_PALETTE: list[Color] = [
    Color(60, 140, 220),
    Color(60, 180, 120),
    Color(220, 120, 60),
    Color(180, 70, 180),
    Color(60, 170, 190),
    Color(200, 170, 60),
]

# HUD
HUD_BG: Color = Color(255, 255, 255, 230)  # semi-transparent white
HUD_TEXT: Color = Color(40, 40, 50)
HUD_BAR_BG: Color = Color(220, 220, 230)
HUD_BAR_FILL: Color = Color(90, 170, 130)
HUD_BAR_HIGH: Color = Color(230, 140, 60)

# Background
BACKGROUND: Color = Color(245, 246, 250)  # off-white (easier on eyes than pure white)
GRID: Color = Color(210, 210, 220)
