"""
 * gameOfLifeInteractive.py
 *
 * Copyright (c) 2026, Jordi-Lluis Figueras
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * OpenAI Codex / ChatGPT 5.4 has been used in the editing of this file.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
"""

"""Interactive Conway's Game of Life board for classroom demonstrations.

Usage:
  - Click on the grid to toggle cells before or during the simulation.
  - Press Run to start the evolution.
  - Press Pause to stop it.
  - Step advances one generation.
  - Clear resets the board.
  - Random fills the board with a random initial condition.
  - Preset buttons load standard patterns such as the Gosper glider gun.

The board uses periodic boundary conditions, so gliders that leave one side of
the grid re-enter on the opposite side.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


nRows = 100
nCols = 100
updateIntervalMs = 100
randomOccupancy = 0.18
rulesText = (
  "Conway's Game of Life rules\n\n"
  "1. Survival: a live cell stays alive if it has 2 or 3 live neighbors.\n"
  "2. Birth: a dead cell becomes alive if it has exactly 3 live neighbors.\n"
  "3. Death by isolation: a live cell dies if it has fewer than 2 live neighbors.\n"
  "4. Death by overcrowding: a live cell dies if it has more than 3 live neighbors.\n\n"
  "This version uses periodic boundary conditions. Cells leaving one edge "
  "re-enter through the opposite edge."
)

gliderPattern = [
  (0, 1),
  (1, 2),
  (2, 0),
  (2, 1),
  (2, 2),
]

pulsarPattern = [
  (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
  (2, 0), (2, 5), (2, 7), (2, 12),
  (3, 0), (3, 5), (3, 7), (3, 12),
  (4, 0), (4, 5), (4, 7), (4, 12),
  (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
  (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
  (8, 0), (8, 5), (8, 7), (8, 12),
  (9, 0), (9, 5), (9, 7), (9, 12),
  (10, 0), (10, 5), (10, 7), (10, 12),
  (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
]

gosperGliderGunPattern = [
  (5, 1), (5, 2), (6, 1), (6, 2),
  (5, 11), (6, 11), (7, 11), (4, 12), (8, 12), (3, 13), (9, 13),
  (3, 14), (9, 14), (6, 15), (4, 16), (8, 16), (5, 17), (6, 17), (7, 17), (6, 18),
  (3, 21), (4, 21), (5, 21), (3, 22), (4, 22), (5, 22), (2, 23), (6, 23),
  (1, 25), (2, 25), (6, 25), (7, 25),
  (3, 35), (4, 35), (3, 36), (4, 36),
]

gliderEaterPattern = [
  (0, 0), (1, 0), (1, 1), (1, 2), (2, 3), (3, 3), (3, 2),
  (15, 14), (15, 15), (15, 16), (16, 14), (17, 15),
]

#Logical gates patterna
logicalAND = [
  #Signal
  (25, 94), (25, 95), (26, 94), (26, 95),
  (25, 85), (26, 85), (27, 85), (24, 84), (28, 84), (23, 83), (29, 83),
  (23, 82), (29, 82), (26, 81), (24, 80), (28, 80), (25, 79), (26, 79), (27, 79), (26, 78),
  (23, 75), (24, 75), (25, 75), (23, 74), (24, 74), (25, 74), (22, 73), (26, 73),
  (21, 71), (22, 71), (26, 71), (27, 71),
  (23, 60), (24, 60), (23, 61), (24, 61),

  #Input A
  (9, 16), (9, 17), (10, 16), (10, 17),
  (9, 26), (10, 26), (11, 26), (8, 27), (12, 27), (7, 28), (13, 28),
  (7, 29), (13, 29), (10, 30), (8, 31), (12, 31), (9, 32), (10, 32), (11, 32), (10, 33),
  (7, 36), (8, 36), (9, 36), (7, 37), (8, 37), (9, 37), (6, 38), (10, 38),
  (5, 40), (6, 40), (10, 40), (11, 40),
  (7, 50), (8, 50), (7, 51), (8, 51),

  #Temporary blocker for input A
  (17, 41), (18, 41), (17, 42), (19, 42),  (19, 43),  (19, 44),  (20, 44),

  #Input B
(24, 1), (24, 2), (25, 1), (25, 2),
(24, 11), (25, 11), (26, 11), (23, 12), (27, 12), (22, 13), (28, 13),
(22, 14), (28, 14), (25, 15), (23, 16), (27, 16), (24, 17), (25, 17), (26, 17), (25, 18),
(22, 21), (23, 21), (24, 21), (22, 22), (23, 22), (24, 22), (21, 23), (25, 23),
(20, 25), (21, 25), (25, 25), (26, 25),
(22, 35), (23, 35), (22, 36), (23, 36),

  #Permanent Blocker
  (87, 14), (87, 15), 
  (88, 13), (88, 15), 
  (89, 13), 
  (90, 12), (90, 13)
]

logicalOR = [
#Signal
(25, 96), (25, 97), (26, 96), (26, 97),
(25, 87), (26, 87), (27, 87), (24, 86),
(28, 86), (23, 85), (29, 85), (23, 84),
(29, 84), (26, 83), (24, 82), (28, 82),
(25, 81), (26, 81), (27, 81), (26, 80),
(23, 77), (24, 77), (25, 77), (23, 76),
(24, 76), (25, 76), (22, 75), (26, 75),
(21, 73), (22, 73), (26, 73), (27, 73),
(23, 62), (24, 62), (23, 63), (24, 63),

  #Input A
(9, 18), (9, 19), (10, 18), (10, 19),
(9, 28), (10, 28), (11, 28), (8, 29),
(12, 29), (7, 30), (13, 30), (7, 31),
(13, 31), (10, 32), (8, 33), (12, 33),
(9, 34), (10, 34), (11, 34), (10, 35),
(7, 38), (8, 38), (9, 38), (7, 39),
(8, 39), (9, 39), (6, 40), (10, 40),
(5, 42), (6, 42), (10, 42), (11, 42),
(7, 52), (8, 52), (7, 53), (8, 53),

#Temporary blocker for input A
(17, 43), (18, 43), (17, 44), (19, 44), (19, 45), (19, 46), (20, 46),

# Input B
(24, 3), (24, 4), (25, 3), (25, 4),
(24, 13), (25, 13), (26, 13), (23, 14),
(27, 14), (22, 15), (28, 15), (22, 16),
(28, 16), (25, 17), (23, 18), (27, 18),
(24, 19), (25, 19), (26, 19), (25, 20),
(22, 23), (23, 23), (24, 23), (22, 24),
(23, 24), (24, 24), (21, 25), (25, 25),
(20, 27), (21, 27), (25, 27), (26, 27),
(22, 37), (23, 37), (22, 38), (23, 38),

#temporary blocker for input B
(32,28), (32,29), (33,28), (34,29), (34,30), (34,31), (35,31),

#signal 2
(54, 1), (54, 2), (55, 1), (55, 2),
(54, 11), (55, 11), (56, 11), (53, 12), (57, 12), (52, 13), (58, 13),
(52, 14), (58, 14), (55, 15), (53, 16), (57, 16), (54, 17), (55, 17), (56, 17), (55, 18),
(52, 21), (53, 21), (54, 21), (52, 22), (53, 22), (54, 22), (51, 23), (55, 23),
(50, 25), (51, 25), (55, 25), (56, 25),
(52, 35), (53, 35), (52, 36), (53, 36),

#blocking for periodic stuff
(7,71), (7,72), (8,71), (9,72), (9,73), (9,74), (10,74),

  #Permanent Blocker
  (87, 14), (87, 15), 
  (88, 13), (88, 15), 
  (89, 13), 
  (90, 12), (90, 13)
]

logicalNOT = [
  #right gosper gun
  (5, 94), (5, 95), (6, 94), (6, 95),
  (5, 85), (6, 85), (7, 85), (4, 84), (8, 84), (3, 83), (9, 83),
  (3, 82), (9, 82), (6, 81), (4, 80), (8, 80), (5, 79), (6, 79), (7, 79), (6, 78),
  (3, 75), (4, 75), (5, 75), (3, 74), (4, 74), (5, 74), (2, 73), (6, 73),
  (1, 71), (2, 71), (6, 71), (7, 71),
  (3, 60), (4, 60), (3, 61), (4, 61),

  (4, 1), (4, 2), (5, 1), (5, 2),
  (4, 11), (5, 11), (6, 11), (3, 12), (7, 12), (2, 13), (8, 13),
  (2, 14), (8, 14), (5, 15), (3, 16), (7, 16), (4, 17), (5, 17), (6, 17), (5, 18),
  (2, 21), (3, 21), (4, 21), (2, 22), (3, 22), (4, 22), (1, 23), (5, 23),
  (0, 25), (1, 25), (5, 25), (6, 25),
  (2, 35), (3, 35), (2, 36), (3, 36),
]

AND_000=[

]

AND_100=[
#Signal
  (25, 94), (25, 95), (26, 94), (26, 95),
  (25, 85), (26, 85), (27, 85), (24, 84), (28, 84), (23, 83), (29, 83),
  (23, 82), (29, 82), (26, 81), (24, 80), (28, 80), (25, 79), (26, 79), (27, 79), (26, 78),
  (23, 75), (24, 75), (25, 75), (23, 74), (24, 74), (25, 74), (22, 73), (26, 73),
  (21, 71), (22, 71), (26, 71), (27, 71),
  (23, 60), (24, 60), (23, 61), (24, 61),

  #Input A
  (9, 16), (9, 17), (10, 16), (10, 17),
  (9, 26), (10, 26), (11, 26), (8, 27), (12, 27), (7, 28), (13, 28),
  (7, 29), (13, 29), (10, 30), (8, 31), (12, 31), (9, 32), (10, 32), (11, 32), (10, 33),
  (7, 36), (8, 36), (9, 36), (7, 37), (8, 37), (9, 37), (6, 38), (10, 38),
  (5, 40), (6, 40), (10, 40), (11, 40),
  (7, 50), (8, 50), (7, 51), (8, 51),

  #Temporary blocker for input A
  (17, 41), (18, 41), (17, 42), (19, 42),  (19, 43),  (19, 44),  (20, 44),

  #Input B
(24, 1), (24, 2), (25, 1), (25, 2),
(24, 11), (25, 11), (26, 11), (23, 12), (27, 12), (22, 13), (28, 13),
(22, 14), (28, 14), (25, 15), (23, 16), (27, 16), (24, 17), (25, 17), (26, 17), (25, 18),
(22, 21), (23, 21), (24, 21), (22, 22), (23, 22), (24, 22), (21, 23), (25, 23),
(20, 25), (21, 25), (25, 25), (26, 25),
(22, 35), (23, 35), (22, 36), (23, 36),

  #Permanent Blocker
  (87, 14), (87, 15), 
  (88, 13), (88, 15), 
  (89, 13), 
  (90, 12), (90, 13)
]

AND_010=[

]

AND_111=[

]

OR_000=[

]

OR_101=[
  
]

OR_011=[
  
]

OR_111=[
  
]

#Done
NOT_10=[
  #right gosper gun
  (5, 94), (5, 95), (6, 94), (6, 95),
  (5, 85), (6, 85), (7, 85), (4, 84), (8, 84), (3, 83), (9, 83),
  (3, 82), (9, 82), (6, 81), (4, 80), (8, 80), (5, 79), (6, 79), (7, 79), (6, 78),
  (3, 75), (4, 75), (5, 75), (3, 74), (4, 74), (5, 74), (2, 73), (6, 73),
  (1, 71), (2, 71), (6, 71), (7, 71),
  (3, 60), (4, 60), (3, 61), (4, 61),

  (4, 1), (4, 2), (5, 1), (5, 2),
  (4, 11), (5, 11), (6, 11), (3, 12), (7, 12), (2, 13), (8, 13),
  (2, 14), (8, 14), (5, 15), (3, 16), (7, 16), (4, 17), (5, 17), (6, 17), (5, 18),
  (2, 21), (3, 21), (4, 21), (2, 22), (3, 22), (4, 22), (1, 23), (5, 23),
  (0, 25), (1, 25), (5, 25), (6, 25),
  (2, 35), (3, 35), (2, 36), (3, 36),
]

NOT_01=[
  #right gosper gun
  (5, 94), (5, 95), (6, 94), (6, 95),
  (5, 85), (6, 85), (7, 85), (4, 84), (8, 84), (3, 83), (9, 83),
  (3, 82), (9, 82), (6, 81), (4, 80), (8, 80), (5, 79), (6, 79), (7, 79), (6, 78),
  (3, 75), (4, 75), (5, 75), (3, 74), (4, 74), (5, 74), (2, 73), (6, 73),
  (1, 71), (2, 71), (6, 71), (7, 71),
  (3, 60), (4, 60), (3, 61), (4, 61),

  (4, 1), (4, 2), (5, 1), (5, 2),
  (4, 11), (5, 11), (6, 11), (3, 12), (7, 12), (2, 13), (8, 13),
  (2, 14), (8, 14), (5, 15), (3, 16), (7, 16), (4, 17), (5, 17), (6, 17), (5, 18),
  (2, 21), (3, 21), (4, 21), (2, 22), (3, 22), (4, 22), (1, 23), (5, 23),
  (0, 25), (1, 25), (5, 25), (6, 25),
  (2, 35), (3, 35), (2, 36), (3, 36),

  #Temporary blocker for input A
  (18, 32), (19, 32), (18, 33), (20, 33), (20, 34), (20, 35), (21, 35)
]

presetPatterns = {
  "Gosper": gosperGliderGunPattern,
  "Glider": gliderPattern,
  "Glider+Eater": gliderEaterPattern,
  "Pulsar": pulsarPattern,
  "AND: |0|0|0|": AND_000,
  "AND: |1|0|0|": AND_100,
  "AND: |0|1|0|": AND_010,
  "AND: |1|1|1|": AND_111,
  "OR: |0|0|0|": OR_000,
  "OR: |1|0|1|": OR_101,
  "OR: |0|1|1|": OR_011,
  "OR: |1|1|1|": OR_111,
  "NOT: |1|0|": NOT_10,
  "NOT: |0|1|": NOT_01,
}

def lifeStep(grid):
  """Return one Game-of-Life update with periodic boundary conditions."""
  neighbors = (
    np.roll(np.roll(grid, 1, axis = 0), 1, axis = 1) +
    np.roll(grid, 1, axis = 0) +
    np.roll(np.roll(grid, 1, axis = 0), -1, axis = 1) +
    np.roll(grid, 1, axis = 1) +
    np.roll(grid, -1, axis = 1) +
    np.roll(np.roll(grid, -1, axis = 0), 1, axis = 1) +
    np.roll(grid, -1, axis = 0) +
    np.roll(np.roll(grid, -1, axis = 0), -1, axis = 1)
  )

  born = (grid == 0) & (neighbors == 3)
  survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
  return (born | survive).astype(int)


class GameOfLifeApp:
  def __init__(self):
    self.grid = np.zeros((nRows, nCols), dtype = int)
    self.isRunning = False
    self.isMouseDown = False
    self.drawValue = 1
    self.rulesFigure = None

    self.fig = plt.figure(figsize = (12, 9))
    self.ax = self.fig.add_axes([0.05, 0.08, 0.70, 0.86])
    self.image = self.ax.imshow(
      self.grid,
      cmap = "binary",
      interpolation = "nearest",
      vmin = 0,
      vmax = 1,
      origin = "upper",
    )

    self.ax.set_title("Conway's Game of Life (periodic 100 x 100 grid)", fontsize = 14)
    self.ax.set_xticks(np.arange(-0.5, nCols, 1), minor = True)
    self.ax.set_yticks(np.arange(-0.5, nRows, 1), minor = True)
    self.ax.grid(which = "minor", color = "lightgray", linewidth = 0.20)
    self.ax.tick_params(which = "both", bottom = False, left = False, labelbottom = False, labelleft = False)

    self.statusText = self.fig.text(
      0.05,
      0.02,
      "Click cells to toggle them. Periodic boundaries are active.",
      fontsize = 11,
    )

    self.timer = self.fig.canvas.new_timer(interval = updateIntervalMs)
    self.timer.add_callback(self.advanceOneStep)

    self.runButton = self._makeButton([0.80, 0.82, 0.1, 0.055], "Run", self.toggleRun)
    self.stepButton = self._makeButton([0.80, 0.75, 0.1, 0.055], "Step", self.stepOnce)
    self.clearButton = self._makeButton([0.80, 0.68, 0.1, 0.055], "Clear", self.clearGrid)
    self.randomButton = self._makeButton([0.80, 0.61, 0.1, 0.055], "Random", self.randomizeGrid)

    self.gosperButton = self._makeButton([0.80, 0.50, 0.1, 0.055], "OR: |0|0|0|", self.loadGosper)
    self.gliderButton = self._makeButton([0.80, 0.43, 0.1, 0.055], "OR: |1|0|1|", self.loadGlider)
    self.gliderEaterButton = self._makeButton([0.80, 0.36, 0.1, 0.055], "OR: |0|1|1|", self.loadGliderEater)
    self.pulsarButton = self._makeButton([0.80, 0.29, 0.1, 0.055], "OR: |1|1|1|", self.loadPulsar)

    self.NOT01Button = self._makeButton([0.80, 0.22, 0.1, 0.055], "NOT: |1|0|", self.loadNOT10)
    self.NOT10Button = self._makeButton([0.68, 0.22, 0.1, 0.055], "NOT: |0|1|", self.loadNOT01)

    self.AND000button = self._makeButton([0.68, 0.50, 0.1, 0.055], "AND: |0|0|0|", self.loadAND000)
    self.AND100Button = self._makeButton([0.68, 0.43, 0.1, 0.055], "AND: |1|0|0|", self.loadAND100)
    self.AND010Button = self._makeButton([0.68, 0.36, 0.1, 0.055], "AND: |0|1|0|", self.loadAND010)
    self.AND111Button = self._makeButton([0.68, 0.29, 0.1, 0.055], "AND: |1|1|1|", self.loadAND111)

    self.fig.canvas.mpl_connect("button_press_event", self.onMousePress)
    self.fig.canvas.mpl_connect("button_release_event", self.onMouseRelease)
    self.fig.canvas.mpl_connect("motion_notify_event", self.onMouseMove)
    self.fig.canvas.mpl_connect("close_event", self.onClose)

  def _makeButton(self, rect, label, callback):
    axis = self.fig.add_axes(rect)
    button = Button(axis, label)
    button.on_clicked(callback)
    return button

  def gridCoordinates(self, event):
    if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
      return None

    col = int(np.floor(event.xdata + 0.5))
    row = int(np.floor(event.ydata + 0.5))

    if row < 0 or row >= nRows or col < 0 or col >= nCols:
      return None

    return row, col

  def refreshDisplay(self):
    self.image.set_data(self.grid)
    self.fig.canvas.draw_idle()

  def updateStatus(self, text):
    self.statusText.set_text(text)
    self.fig.canvas.draw_idle()

  def paintCell(self, row, col, value):
    if self.grid[row, col] != value:
      self.grid[row, col] = value
      self.refreshDisplay()

  def placePattern(self, pattern, anchorRow = 0, anchorCol = 0):
    # Clear the board
    self.grid.fill(0)
    
    # Place the pattern exactly at its defined coordinates
    for rowOffset, colOffset in pattern:
      row = (anchorRow + rowOffset) % nRows
      col = (anchorCol + colOffset) % nCols
      self.grid[row, col] = 1

    self.refreshDisplay()

  def loadPreset(self, name):
    if self.isRunning:
      self.toggleRun(None)

    self.placePattern(presetPatterns[name])
    self.updateStatus(f"Loaded preset: {name}.")

  def onMousePress(self, event):
    coordinates = self.gridCoordinates(event)
    if coordinates is None:
      return

    row, col = coordinates
    self.isMouseDown = True
    self.drawValue = 1 - self.grid[row, col]
    self.paintCell(row, col, self.drawValue)

  def onMouseMove(self, event):
    if not self.isMouseDown:
      return

    coordinates = self.gridCoordinates(event)
    if coordinates is None:
      return

    row, col = coordinates
    self.paintCell(row, col, self.drawValue)

  def onMouseRelease(self, _event):
    self.isMouseDown = False

  def toggleRun(self, _event):
    if self.isRunning:
      self.isRunning = False
      self.timer.stop()
      self.runButton.label.set_text("Run")
      self.updateStatus("Simulation paused. You can keep editing the grid.")
    else:
      self.isRunning = True
      self.timer.start()
      self.runButton.label.set_text("Pause")
      self.updateStatus("Simulation running with periodic boundaries.")
      self.fig.canvas.draw_idle()

  def stepOnce(self, _event):
    if self.isRunning:
      self.toggleRun(None)

    self.advanceOneStep()
    self.updateStatus("Advanced one generation.")

  def clearGrid(self, _event):
    if self.isRunning:
      self.toggleRun(None)

    self.grid.fill(0)
    self.refreshDisplay()
    self.updateStatus("Grid cleared.")

  def randomizeGrid(self, _event):
    if self.isRunning:
      self.toggleRun(None)

    self.grid = (np.random.random((nRows, nCols)) < randomOccupancy).astype(int)
    self.refreshDisplay()
    self.updateStatus("Random initial condition loaded.")

  def loadGosper(self, _event):
    self.loadPreset("Gosper")

  def loadGlider(self, _event):
    self.loadPreset("Glider")

  def loadGliderEater(self, _event):
    self.loadPreset("Glider+Eater")

  def loadPulsar(self, _event):
    self.loadPreset("Pulsar")

  def loadAND000(self, _event):
    self.loadPreset("AND: |0|0|0|")  
  
  def loadAND100(self, _event):
    self.loadPreset("AND: |1|0|0|")  

  def loadAND010(self, _event):
    self.loadPreset("AND: |0|1|0|")  

  def loadAND111(self, _event):
    self.loadPreset("AND: |1|1|1|")  

  def loadOR000(self, _event):
    self.loadPreset("OR: |0|0|0|")  
  
  def loadOR100(self, _event):
    self.loadPreset("OR: |1|0|1|")  

  def loadOR010(self, _event):
    self.loadPreset("OR: |0|1|1|")  

  def loadAOR111(self, _event):
    self.loadPreset("OR: |1|1|1|")  

  def loadNOT10(self, _event):
    self.loadPreset("NOT: |0|1|")

  def loadNOT01(self, _event):
    self.loadPreset("NOT: |1|0|")

  def showRules(self, _event):
    if self.rulesFigure is not None and plt.fignum_exists(self.rulesFigure.number):
      self.rulesFigure.canvas.manager.show()
      self.rulesFigure.canvas.draw_idle()
      return

    self.rulesFigure = plt.figure(figsize = (6.8, 4.0))
    self.rulesFigure.suptitle("Game of Life Rules", fontsize = 13)
    rulesAxis = self.rulesFigure.add_axes([0.06, 0.08, 0.88, 0.8])
    rulesAxis.axis("off")
    rulesAxis.text(0.0, 1.0, rulesText, va = "top", fontsize = 11, wrap = True)
    self.rulesFigure.canvas.draw_idle()

  def advanceOneStep(self):
    self.grid = lifeStep(self.grid)
    self.refreshDisplay()

  def onClose(self, _event):
    self.timer.stop()


def main():
  GameOfLifeApp()
  plt.show()


if __name__ == "__main__":
  main()
