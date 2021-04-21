class Controller:
    def __init__(self, P=0.0, D=0.0, set_point=0, rate=1.0):
        """
        Parameters
        ----------
        P : float, optional
            The proportional gain (default is 0.0)
        D : float, optional
            The derivative gain (default is 0.0)
        set_point : int, optional
            The one-dimensional point to track (default is 0)
        rate: float, optional
            The update rate of the controller in Hz (default is 1.0)
        """
        self.Kp = P
        self.Kd = D * rate
        self.set_point = set_point # reference (desired value)
        self.previous_error = 0

    def update(self, current_value):
        """ Calculates the control input given state current_value """
        error = self.set_point - current_value
        P_term = self.Kp * error
        D_term = self.Kd * (error - self.previous_error)
        self.previous_error = error
        return P_term + D_term

    def setPoint(self, set_point):
        """ Updates the one-dimensional point to track """
        self.set_point = set_point
        self.previous_error = 0

    def setPD(self, P=0.0, D=0.0):
        """ Updates the proportional and derivative gains """
        self.Kp = P
        self.Kd = D
        