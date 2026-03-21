from pygame import Color

# Node states
NODE_UNVISITED: Color = Color(220, 80, 60)  # warm red — orphan / unassigned
NODE_ASSIGNED: Color = Color(60, 160, 120)  # teal — in a planned route
NODE_DEPOT: Color = Color(80, 80, 200)  # blue — depot
NODE_VISITED: Color = Color(160, 160, 160)  # gray — already collected

# Trucks
TRUCK_ACTIVE: Color = Color(240, 200, 60)  # amber
TRUCK_BROKEN: Color = Color(120, 120, 120)  # gray
TRUCK_RECOVERING: Color = Color(180, 140, 220)  # purple

# Route lines — one per truck, cycle through these
ROUTE_PALETTE: list[Color] = [
    Color(80, 160, 240),
    Color(80, 220, 140),
    Color(240, 140, 80),
    Color(200, 80, 200),
    Color(80, 200, 220),
    Color(220, 200, 80),
]

# HUD
HUD_BG: Color = Color(20, 20, 30, 200)  # RGBA, semi-transparent panel
HUD_TEXT: Color = Color(220, 220, 220)
HUD_BAR_BG: Color = Color(60, 60, 70)
HUD_BAR_FILL: Color = Color(80, 180, 120)
HUD_BAR_HIGH: Color = Color(240, 160, 60)  # membership bar when value > 0.7

# Background
BACKGROUND: Color = Color(15, 15, 25)
GRID: Color = Color(30, 30, 45)
