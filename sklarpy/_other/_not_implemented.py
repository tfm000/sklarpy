__all__ = ['NotImplemented']


class NotImplemented:
    def _not_implemented(self, func_name):
        raise NotImplementedError(f"{func_name} not implemented for {self.name}")
