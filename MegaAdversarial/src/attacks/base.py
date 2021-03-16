from .attack import Attack, Attack2Ensemble


class Clean(Attack):
    def forward(self, x, y, kwargs):
        return x, y

class Clean2Ensemble(Attack2Ensemble):
    def forward(self, x, y, kwargs_arr):
        return x, y
