from typing import List


class BatteryCell:
    def __init__(self, voltage, capacity, internal_resistance, soc=1.0):
        """
        Initialize a battery model with given voltage, capacity, and internal resistance.
        
        :param voltage: Nominal voltage of the battery (V)
        :param capacity: Capacity of the battery (Ah or mAh)
        :param internal_resistance: Internal resistance of the battery (ohms)
        :param soc: State of Charge, a value between 0 and 1. Default is None which means undefined.
        """
        self.voltage = voltage
        self.capacity = capacity
        self.internal_resistance = internal_resistance
        self.soc = soc  # assuming 1.0 (or 100%) by default

    def voltage_at_current(self, current):
        return self.voltage - current * self.internal_resistance
    
    def power_at_current(self, current):
        return current**2 * self.internal_resistance
    
    def get_voltage_drop(self, current):
        """
        Calculate the voltage drop due to internal resistance when a current is drawn.

        :param current: Current drawn from the battery (A)
        :return: Voltage drop (V)
        """
        return self.internal_resistance * current


class IRBatteryModel(BatteryCell):
    def discharge(self, current, hours):
        """
        Discharge the battery by a given current for specified hours. 
        Decrease SOC accordingly. For simplicity, we assume linear discharge.
        
        :param current: Discharge current (A)
        :param hours: Duration for which the battery is discharged (hours)
        """
        consumed_capacity = current * hours
        self.soc = max(0, self.soc - consumed_capacity / self.capacity)

    def charge(self, current, hours):
        """
        Charge the battery by a given current for specified hours. 
        Increase SOC accordingly. For simplicity, we assume linear charge.
        
        :param current: Charge current (A)
        :param hours: Duration for which the battery is charged (hours)
        """
        added_capacity = current * hours
        self.soc = min(1, self.soc + added_capacity / self.capacity)

    def __str__(self):
        return f"IRBatteryModel(Voltage: {self.voltage}V, Capacity: {self.capacity}Ah, IR: {self.internal_resistance}Ω, SOC: {self.soc if isinstance(self.soc, str) else round(self.soc*100, 2)}%)"

class ParallelGroup:
    def __init__(self, cell: BatteryCell, num_cells: int):
        self.cell = cell
        self.num_cells = num_cells

    def voltage_at_current(self, current):
        return self.cell.voltage_at_current(current/self.num_cells)
    
    def power_at_current(self, current):
        return self.cell.power_at_current(current/self.num_cells) * self.num_cells
    
    def get_voltage_drop(self, current):
        return self.cell.get_voltage_drop(current/self.num_cells)
    
    def get_resistance(self):
        return 1 / ((1/self.cell.internal_resistance) * self.num_cells)

class BatteryPack:
    def __init__(self, groups: List[ParallelGroup]):
        self.groups = groups

    def total_voltage(self):
        return sum(group.voltage_at_current(0) for group in self.groups)

    def total_internal_resistance(self):
        # Resistance of cells in parallel gets divided by the number of cells
        return sum(group.get_resistance() for group in self.groups)

    def power_loss_at_current(self, current):
        # P = I^2 * R
        return sum(group.power_at_current(current) for group in self.groups)

    def output_voltage_at_current(self, current):
        return sum(group.voltage_at_current(current) for group in self.groups)

    def output_power_at_current(self, current):
        return self.output_voltage_at_current(current) * current
    
    def get_pack_at_power(self, power):
        # calculate the voltage, current, and power loss of the pack at a given output power (W)
        # P = IV
        # iteratively solve for I and V
        # first we will start at the max voltage and work our way down
        # we will use the bisection method to find the voltage
        i = 0
        v = self.total_voltage()
        current = power / v
        last_current = 0
        highest_power = 0
        while abs(current - last_current) > 0.001 and i < 100:
            v = self.output_voltage_at_current(current)
            new_power = v * current
            if new_power > power and current > last_current:
                pass
            if new_power > highest_power:
                highest_power = new_power
            # print(f"Current: {current:.2f} {last_current:.2f}, Voltage: {v:.3f}, Power: {power:.3f} {new_power:.3f}")
            last_current = current
            current = power / v
            i += 1
        return v, current, self.power_loss_at_current(current)