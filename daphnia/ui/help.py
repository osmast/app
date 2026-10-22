"""In-app help: the calibration procedure behind a "?" button."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout

#: The exact procedure, as shown in the dialog.
CALIBRATION_HELP = """
<h2>How to set up calibration</h2>
<p>The reference target is a sheet of <b>printed grid paper</b> - 1 mm squares by
default. Panel 1 names the optical chain, panel 2 takes one photo of the paper and
measures how many pixels one square spans, across the whole frame, and panel 3 shows the
resulting scale; <i>Store this scale</i> writes it back onto the chain in panel 1. The
whole daily act is: one photo, one press, one result, one stored record.</p>
<ol>
<li><b>Name the chain.</b> One name is one optical chain: every calibration saved under
the same name shares one history, and the line under the name says what is in force for
it now - with <i>RE-VERIFY</i> once it is older than 12 hours. If you work through two
objectives, use two names.</li>
<li><b>Mount the phone.</b> Clamp it over the eyepiece with the adapter so the wide
camera looks straight into the eyepiece - camera zoom exactly <b>1.0x</b>, focus locked
to <b>infinity</b>. Do not move it again during the session: a change to the zoom, the
focus or the objective changes the scale.</li>
<li><b>Set the printed square</b> in panel 2 to the paper you actually have: <b>1 mm</b>,
or 0.5 mm / 0.1 mm if you need finer optics calibrated. The detector measures the period
in pixels; this number is what turns it into a scale.</li>
<li><b>Put the grid paper</b> under the objective - at the same height the specimen will
be, printed side up, as flat and as perpendicular to the lens as you can, filling the
field.</li>
<li><b>Take one photo</b> in panel 2: press <i>Use live frame</i>, or <i>Open photo...</i>
for any file you already have (an iPhone <i>.HEIC</i> is converted for you).</li>
<li><b>Press <i>Measure grid</i>.</b> It finds the circular paper area inside the
eyepiece's field stop, straightens the tilted print and measures the period off narrow
bands of the photo. Both are drawn <b>over the photo</b> - the lattice in amber and the
circle it was measured inside - so you can see what was measured rather than trust a
number.</li>
<li><b>Read the verdict.</b> <i>MEASURED</i> means the field's own readings agree to
within the print's tolerance. <i>WEAK - the field moves more than the print</i> means the
scale really does change across the frame: it is right where it was measured and less
right towards the edge, so keep the specimen central. <i>REFUSED</i> means there is no
lattice in that frame - check the paper is flat, in focus and filling the field; the
reason is in the line's tooltip.</li>
<li><b>The drift is not the accuracy.</b> The drift line is how much the scale moves from
the centre of the field to its edge, measured from the same bands; the accuracy is how
well the centre was measured - including the print's own half percent on the grid route.
Neither is the <i>lens distortion</i>: a correction for that would be a model fitted for
this chain, none exists, and nothing here rewrites a frame - which is why panel 1 says
"unmeasured and not corrected", and why you can ignore it while the specimen stays near
the centre.</li>
<li><b>Measuring a span by hand is the fallback</b> (it sits below the rule in panel 2) -
click the first and the last line of N squares. Both routes set the <b>one</b> scale for
the whole frame and neither corrects the lens; the difference is precision: the two
clicks each carry about a pixel of error and a span's accuracy does <b>not</b> include the
print's tolerance, while the detector uses every square in view and does. Use it when the
paper is not fully in view.</li>
<li><b>Already know the scale?</b> Tick <i>I know the scale already</i> in panel 3 and type
it in px/um. It overrides the measurement, it is stored as your claim with the accuracy
you state for it, and it is never presented as a measurement - it has no proof photo.</li>
<li><b>Store it.</b> <i>Store this scale</i> - the button in panel 3, and the same entry in
the window's <i>Calibration</i> menu - stores a dated calibration against the chain in
panel 1: what the scale was, when it was verified, and how accurate it was. Measurements
already taken keep the calibration they were taken under.</li>
</ol>
<p><b>Redo it whenever the optics could have changed:</b> at the start of every session,
and after any refocus, objective change or knock to the adapter. The bar at the bottom of
every window shows the scale, its accuracy and its age, and flags when it is time to
re-verify.</p>
<p><b>One honest limitation:</b> printed paper is not a length standard. Its own spacing
is only good to roughly half a percent, and that becomes the floor under every measurement
you take with it. If you ever need better, verify this sheet once against a glass stage
micrometer and record the difference.</p>
"""


class CalibrationHelpDialog(QDialog):
    """Modal "how to set up calibration" help."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to set up calibration")
        self.setModal(True)
        self.resize(640, 640)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(CALIBRATION_HELP)
        layout.addWidget(browser)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close, alignment=Qt.AlignRight)
