import numpy as np
import cv2


class TrafficLightDetection:
    """
    Full-frame circular-blob traffic-light detector.

    For each color class (red / yellow / green) we:
      1. Build an HSV mask with cv2.inRange (red uses two ranges because its
         hue wraps the 0/180 boundary).
      2. Clean the mask with open+close morphology to kill salt-and-pepper
         noise and seal small holes.
      3. findContours and keep the blob whose shape is most disk-like:
         large enough, high circularity, and high fill ratio inside its
         minimum enclosing circle.
      4. The color with the largest accepted circular blob wins.
    """

    # ------------------------------------------------------------------
    # Color classification  (OpenCV HSV: H in [0,179])
    #
    # Every sufficiently bright/saturated pixel is assigned to whichever
    # hue center it's closest to (in circular distance). Yellow and green
    # are no longer separated by a hard cutoff — the implicit boundary is
    # the midpoint between their centers (~42), so bulbs in the yellow/
    # green transition zone are classified by which center they lean
    # toward rather than by an arbitrary fixed bound.
    # ------------------------------------------------------------------
    HUE_CENTERS = {
        "red":     0,   # red wraps around 0/180; circular distance handles it
        "yellow": 40,   # calibrated from the on-robot camera: bulb hue ~40
        "green":  65,
    }
    MAX_HUE_DIST = 22   # ignore pixels farther than this from every center
    MIN_S = 65          # min saturation — bulb halo measured ~70-115
    MIN_V = 85          # min value      — bulb halo measured ~95-130

    # ------------------------------------------------------------------
    # "White-out" rescue
    #
    # A bright LED bulb often saturates the camera sensor in its center,
    # producing a near-white core surrounded by a colored halo. Those white
    # pixels fail the MIN_S floor and would normally fall in no color mask,
    # leaving only a thin ring that's not a great circle candidate. This
    # rescue step finds those white cores and attaches each one to whichever
    # color's mask shows up most in the halo around it.
    # ------------------------------------------------------------------
    WHITE_V_MIN = 200      # only very bright pixels qualify as "blown out"
    WHITE_S_MAX = 80       # ...and only if they're nearly desaturated
    HALO_DILATE_PX = 7     # how far around the white core to sample halo color
    HALO_OVERLAP_MIN = 20  # min pixels of matching color halo to claim a core

    # ------------------------------------------------------------------
    # Shape / size acceptance thresholds
    # ------------------------------------------------------------------
    MIN_AREA = 60            # smallest blob accepted, in pixels
    MIN_CIRCULARITY = 0.60   # 4*pi*A / P^2  (perfect circle = 1.0)
    MIN_FILL_RATIO = 0.65    # blob_area / minEnclosingCircle_area

    # ------------------------------------------------------------------
    # Debug rendering — every tile is letterboxed to this exact size, so
    # the window never resizes / flickers between frames.
    # ------------------------------------------------------------------
    DEBUG_TILE_W = 320
    DEBUG_TILE_H = 240
    DEBUG_WINDOW = "Traffic Light Debug"

    BGR_FOR = {
        "red":    (0,   0, 255),
        "yellow": (0, 255, 255),
        "green":  (0, 255,   0),
        "none":   (200, 200, 200),
    }

    def __init__(self, debug=False):
        self.debug = debug
        self._debug_window_ready = False
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (5, 5))
        self._dbg = {}
        # Latest composite debug image (BGR ndarray). Populated on every
        # detect_state() call so external consumers (e.g. a ROS publisher)
        # can grab the same picture the local cv2.imshow window shows.
        self.last_debug_frame = None

    # ==================================================================
    # Pipeline
    # ==================================================================

    def detect_state(self, image):
        """Return 'red', 'yellow', 'green', or 'none'."""
        if image is None:
            return "none"

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        masks = self._classify_pixels(hsv)
        detections = {c: self._best_circle(m) for c, m in masks.items()}

        # Pick the color whose best circle has the largest area.
        best_color, (best_area, _) = max(
            detections.items(), key=lambda kv: kv[1][0]
        )
        if best_area <= 0:
            best_color = "none"

        # Always populate the debug intermediates and render the composite,
        # so the node can republish it as an Image topic even when the local
        # cv2.imshow window is turned off.
        self._dbg = {
            "input": image,
            "hsv": hsv,
            "masks": masks,
            "detections": detections,
            "state": best_color,
        }
        self.last_debug_frame = self._render_debug_frame()
        if self.debug:
            self._show_debug_window(self.last_debug_frame)

        return best_color

    # ==================================================================
    # Pixel classification
    # ==================================================================

    def _classify_pixels(self, hsv):
        """
        Build one binary mask per color via nearest-hue-center assignment.

        Every pixel that clears the saturation/value floor is tagged with
        the color whose HUE_CENTERS entry is the closest in circular hue
        distance, provided that distance is <= MAX_HUE_DIST. Pixels too
        far from every center (e.g. blue, cyan, gray) end up in no mask.

        The masks are mutually exclusive by construction — there is no
        yellow/green overlap zone.
        """
        h_chan, s_chan, v_chan = cv2.split(hsv)
        bright = (s_chan >= self.MIN_S) & (v_chan >= self.MIN_V)

        h = h_chan.astype(np.int16)
        color_names = list(self.HUE_CENTERS.keys())
        centers = list(self.HUE_CENTERS.values())
        # Circular hue distance per pixel, per color (stacked along axis -1).
        distances = np.stack([
            np.minimum(np.abs(h - c), 180 - np.abs(h - c)) for c in centers
        ], axis=-1)
        nearest = np.argmin(distances, axis=-1)
        nearest_dist = np.take_along_axis(
            distances, nearest[..., None], axis=-1
        ).squeeze(-1)

        masks = {}
        for i, name in enumerate(color_names):
            m = (bright
                 & (nearest == i)
                 & (nearest_dist <= self.MAX_HUE_DIST)).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self._morph_kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._morph_kernel)
            masks[name] = m

        self._rescue_blown_cores(s_chan, v_chan, masks)
        return masks

    def _rescue_blown_cores(self, s_chan, v_chan, masks):
        """
        Find bright near-white blobs (LED cores blown out by the camera) and
        merge each one into whichever color mask wraps it most. Mutates
        `masks` in place. Skips noisy or non-round white blobs to avoid
        attaching, say, a piece of paper to the red mask.
        """
        bright_white = ((v_chan >= self.WHITE_V_MIN)
                        & (s_chan <= self.WHITE_S_MAX)).astype(np.uint8) * 255
        bright_white = cv2.morphologyEx(bright_white, cv2.MORPH_OPEN,
                                        self._morph_kernel)
        # Reset debug buffers every call so stale shapes from a previous
        # frame can't leak through (e.g. when image size changes).
        self._dbg_white_mask = bright_white
        self._dbg_rescued_mask = np.zeros_like(bright_white)
        if cv2.countNonZero(bright_white) == 0:
            return

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bright_white, connectivity=8
        )
        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.HALO_DILATE_PX * 2 + 1, self.HALO_DILATE_PX * 2 + 1),
        )

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.MIN_AREA:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            if not self._is_roundish(component):
                continue
            ring = cv2.subtract(cv2.dilate(component, halo_kernel), component)

            overlaps = {
                c: int(cv2.countNonZero(cv2.bitwise_and(ring, m)))
                for c, m in masks.items()
            }
            best_color = max(overlaps, key=lambda c: overlaps[c])
            if overlaps[best_color] < self.HALO_OVERLAP_MIN:
                continue
            masks[best_color] = cv2.bitwise_or(masks[best_color], component)
            self._dbg_rescued_mask = cv2.bitwise_or(
                self._dbg_rescued_mask, component
            )

    def _is_roundish(self, mask):
        """Quick circularity check on a single-blob mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0 or area <= 0:
            return False
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        return circularity >= self.MIN_CIRCULARITY

    # ==================================================================
    # Circle picking
    # ==================================================================

    def _best_circle(self, mask):
        """
        Return (area, contour) of the most disk-like blob in `mask`,
        or (0, None) if nothing qualifies.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_area = 0
        best_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.MIN_AREA:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue
            (_, _), radius = cv2.minEnclosingCircle(c)
            if radius <= 0:
                continue
            fill = area / (np.pi * radius * radius)
            if fill < self.MIN_FILL_RATIO:
                continue
            if area > best_area:
                best_area = area
                best_contour = c
        return best_area, best_contour

    # ==================================================================
    # Debug rendering
    # ==================================================================

    @staticmethod
    def _label(img, text, color=(255, 255, 255)):
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _colorize(mask, bgr_color):
        canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
        canvas[mask > 0] = bgr_color
        return canvas

    @classmethod
    def _tile(cls, img, label=None, label_color=(255, 255, 255)):
        """Letterbox `img` into a fixed DEBUG_TILE_W x DEBUG_TILE_H tile."""
        W, H = cls.DEBUG_TILE_W, cls.DEBUG_TILE_H
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if (img is not None and img.ndim >= 2
                and img.shape[0] >= 2 and img.shape[1] >= 2):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = np.ascontiguousarray(img)
            ih, iw = img.shape[:2]
            scale = min(W / iw, H / ih)
            new_w = max(1, int(round(iw * scale)))
            new_h = max(1, int(round(ih * scale)))
            resized = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            x_off = (W - new_w) // 2
            y_off = (H - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        if label is not None:
            cls._label(canvas, label, label_color)
        return canvas

    def _draw_circle_overlay(self, image):
        """Outline every accepted circle on a copy of the input."""
        out = image.copy()
        for color, (_, contour) in self._dbg["detections"].items():
            if contour is None:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(out, (int(cx), int(cy)), int(radius),
                       self.BGR_FOR[color], 2)
        return out

    def _render_debug_frame(self):
        """
        Build the fixed 2x3 pipeline composite and return it as a BGR ndarray.

          Row 1 │ 1. camera + circles │ 2. white-out rescue │ 3. winner crop
          Row 2 │ 4. red mask         │ 5. yellow mask      │ 6. green mask
        """
        state = self._dbg["state"]
        masks = self._dbg["masks"]
        detections = self._dbg["detections"]
        state_color = self.BGR_FOR.get(state, (255, 255, 255))

        # --- Row 1 ----------------------------------------------------
        cam_tile = self._tile(
            self._draw_circle_overlay(self._dbg["input"]),
            f"1. camera  state={state}",
            state_color,
        )

        # Tile 2: white-out rescue. Gray = bright near-white pixels detected;
        # bright white = cores that were successfully attached to a color halo.
        white_mask = getattr(self, "_dbg_white_mask", None)
        rescued_mask = getattr(self, "_dbg_rescued_mask", None)
        if white_mask is not None:
            white_viz = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR) // 2
            if rescued_mask is not None:
                white_viz[rescued_mask > 0] = (255, 255, 255)
            white_tile = self._tile(white_viz, "2. white-out rescue")
        else:
            white_tile = self._tile(None, "2. white-out rescue")

        if state != "none":
            area, contour = detections[state]
            isolated = cv2.bitwise_and(self._dbg["input"],
                                       self._dbg["input"],
                                       mask=masks[state])
            if contour is not None:
                cv2.drawContours(isolated, [contour], -1,
                                 (255, 255, 255), 2)
            winner_tile = self._tile(
                isolated,
                f"3. winner: {state}  area={int(area)}",
                state_color,
            )
        else:
            winner_tile = self._tile(None, "3. no detection")

        row1 = np.hstack([cam_tile, white_tile, winner_tile])

        # --- Row 2 ----------------------------------------------------
        panels = []
        for i, color in enumerate(("red", "yellow", "green"), start=4):
            area, contour = detections[color]
            viz = self._colorize(masks[color], self.BGR_FOR[color])
            if contour is not None:
                cv2.drawContours(viz, [contour], -1, (255, 255, 255), 2)
            panels.append(self._tile(
                viz,
                f"{i}. {color} mask  area={int(area)}",
                self.BGR_FOR[color],
            ))
        row2 = np.hstack(panels)

        return np.vstack([row1, row2])

    def _show_debug_window(self, frame):
        """Display `frame` in a stable, fixed-size OpenCV window."""
        if not self._debug_window_ready:
            cv2.namedWindow(self.DEBUG_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.DEBUG_WINDOW,
                             self.DEBUG_TILE_W * 3,
                             self.DEBUG_TILE_H * 2)
            self._debug_window_ready = True
        cv2.imshow(self.DEBUG_WINDOW, frame)
        cv2.waitKey(1)
