import math

_ALIGN_TABLES = {
    "QB3":  [(.20,8.80,1.87,1.55),(.25,6.13,1.67,1.45),(.30,4.42,1.50,1.36),
             (.35,3.22,1.35,1.27),(.40,2.35,1.22,1.19),(.45,1.70,1.10,1.10),(.50,1.20,1.00,1.00)],
    "SBB4": [(.20,16.32,2.09,1.00),(.25,10.04,1.85,1.00),(.30,6.57,1.64,1.00),
             (.35,4.50,1.46,1.00),(.40,3.18,1.30,1.00),(.45,2.29,1.16,1.00),(.50,1.67,1.04,1.00)],
    "B4":   [(.20,6.97,1.56,1.56),(.25,4.47,1.41,1.41),(.30,3.05,1.28,1.28),
             (.35,2.17,1.16,1.16),(.40,1.58,1.07,1.07),(.45,1.18,0.98,0.98),(.50,0.89,0.91,0.91)],
}

class AlignmentEngine:
    def __init__(self, fs: float, qts: float, vas: float):
        self.fs = float(fs)
        self.qts = float(qts)
        self.vas = float(vas)

    def _interp_align(self, align: str):
        qts = self.qts
        if qts < 0.20:
            qts = 0.20
        elif qts > 0.50:
            qts = 0.50

        rows = _ALIGN_TABLES.get(align)
        if not rows:
            return None

        # Interpolación lineal simple (idéntica a MATCH/INDEX en Excel de SpeakerLab)
        for i in range(len(rows)-1):
            q0, a0, h0, f0 = rows[i]
            q1, a1, h1, f1 = rows[i+1]
            if q0 <= qts <= q1:
                t = (qts - q0) / (q1 - q0)
                alpha = a0 + t * (a1 - a0)
                h_fb = h0 + t * (h1 - h0)
                f3_ratio = f0 + t * (f1 - f0)
                return alpha, h_fb, f3_ratio
        
        # Fallback al último valor si qts es topado a 0.50
        return rows[-1][1], rows[-1][2], rows[-1][3]

    def _calc(self, align: str):
        params = self._interp_align(align)
        if not params:
            return {"vb": 0.0, "fb": 0.0, "f3": 0.0}
        
        alpha, h_fb, f3_ratio = params
        
        # Matemáticas forzadas con ROUND(..., 1) según el Excel: RESULTADOS!B5 y B6
        vb = round(float(self.vas / alpha), 1)
        fb = round(float(h_fb * self.fs), 1)
        f3 = round(float(f3_ratio * self.fs), 1)
        
        return {"vb": vb, "fb": fb, "f3": f3}

    def get_all_alignments(self):
        return {
            "QB3": self.calculate_qb3(),
            "SBB4": self.calculate_sbb4(),
            "B4": self.calculate_b4()
        }

    def calculate_qb3(self):
        return self._calc("QB3")

    def calculate_sbb4(self):
        return self._calc("SBB4")

    def calculate_b4(self):
        return self._calc("B4")