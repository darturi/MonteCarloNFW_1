import copy
import math
import numpy as np
import scipy.constants as const

class SimpleSurface:
    def __init__(self, temperature, acc_barrier_diff, acc_barrier_desorption, E_d, E_r, surface_width, per_site_flux, seed=12345):
        self.T = temperature  # Kelvin

        self.boltzmann = 8.617269 * math.pow(10, -5)  # in eVK^-1

        self.A_d = acc_barrier_diff  # Activation barrier for diffusion (in sec^-1)
        self.E_d = E_d  # In eV

        self.diffusion_rate = self.A_d * np.exp(-self.E_d / (self.T * self.boltzmann))

        self.A_r = acc_barrier_desorption  # Activation barrier for desorption (in sec^-1)
        self.E_r = E_r   # In eV

        self.desorption_rate = self.A_r * np.exp(-self.E_r / (self.T * self.boltzmann))

        self.surface_width = surface_width
        self.surface_array = [[0 for _ in range(self.surface_width)]]

        # F / X, where
        #   * F is deposition flux in length^-2 sec^-1 and
        #   * X is the aerial density of surface sites in units of length^-2
        #   * F / X is the per site flux in sec^-1
        self.per_site_flux = per_site_flux

        self.rng = np.random.default_rng(seed)

        self.time = 0

        # TODO: Convert to writing to a JSON
        # hist is structured as a list of dicts of the following form:
        # {
        #   start_time: x,
        #   end_time: y,
        #   normalized_state_rates: [...],
        #   grid_after_step: [...],
        #   random_num_chosen: z
        #   action_taken: (name, index) # TODO
        # }
        self.hist = []

    def calculate_possible_state_change_rates(self):
        # (rate, (func, args))
        state_change_rates = []
        rate_sum = 0

        for row in range(len(self.surface_array)):
            for col in range(self.surface_width):

                if self.surface_array[row][col] == 0:
                    state_change_rates.append(
                        [self.per_site_flux, (self.deposition_event, [(row, col)])]
                    )
                    rate_sum += self.per_site_flux
                elif self.surface_array[row][col] == 1:
                    # Check if left open, if so add left diffuse possibility
                    if col > 0 and self.surface_array[row][col-1] <= 0:
                        state_change_rates.append(
                            [self.diffusion_rate, (self.diffusion_event, [(row, col), (row, col-1)])]
                        )
                        rate_sum += self.diffusion_rate
                    # Check if right open, if so add right diffuse possibility
                    if col < self.surface_width - 1 and self.surface_array[row][col+1] <= 0:
                        state_change_rates.append(
                            [self.diffusion_rate, (self.diffusion_event, [(row, col), (row, col + 1)])]
                        )
                        rate_sum += self.diffusion_rate
                    # Desorption possibility
                    state_change_rates.append(
                        [self.desorption_rate, (self.desorption_event, [(row, col)])]
                    )
                    rate_sum += self.desorption_rate

        # Normalize probabilities
        # for elem in state_change_rates:
        #     elem[0] /= rate_sum

        return rate_sum, state_change_rates

    def step_atomic(self):
        rate_sum, state_rates = self.calculate_possible_state_change_rates()

        # Normalize probabilities
        for elem in state_rates:
            elem[0] /= rate_sum

        r_float = self.rng.random()

        hist_entry = {"start_time": self.time}
        self.time += self.compute_time_delta(r_float, rate_sum)
        hist_entry["end_time"] = self.time
        hist_entry["normalized_state_rates"] = state_rates
        hist_entry["random_num_chosen"] = r_float

        curr = 0
        curr_action_index = 0
        while curr + state_rates[curr_action_index][0] < r_float:
            curr += state_rates[curr_action_index][0]
            curr_action_index += 1

        chosen_action = state_rates[curr_action_index][1][0]
        args = state_rates[curr_action_index][1][1]

        chosen_action(args)

        if chosen_action == self.diffusion_event:
            hist_entry["action_taken"] = (f"diffusion_event: [{args[0][0]}, {args[0][1]}] ==> [{args[1][0]}, {args[1][1]}]", curr_action_index)
        elif chosen_action == self.desorption_event:
            hist_entry["action_taken"] = (f"desorption_event: [{args[0][0]}, {args[0][1]}]", curr_action_index)
        else:
            hist_entry["action_taken"] = (f"deposition_event: [{args[0][0]}, {args[0][1]}]", curr_action_index)


        hist_entry["grid_after_step"] = [row[:] for row in self.surface_array]

        self.hist.append(hist_entry)

    def take_n_steps(self, n):
        for _ in range(n):
            self.step_atomic()

    def compute_time_delta(self, r_float, rate_sum):
        return -math.log(r_float) / rate_sum


    # -----------------------------------------------
    # ACTIONS
    def diffusion_event(self, args):
        start, end = args
        s_row, s_col = start
        self.surface_array[s_row][s_col] = 0
        # self.surface_array[end[0]][end[1]] = 1

        self.surface_array[s_row + 1][s_col] = -1
        # self.surface_array[end[0] + 1][end[1]] = 0

        # Upon diffusion to a place above empty space, fall
        new_row = end[0]
        while new_row > 0 and self.surface_array[new_row - 1][end[1]] != 1:
            new_row -= 1

        self.surface_array[new_row][end[1]] = 1
        self.surface_array[new_row + 1][end[1]] = 0

        # If there is stuff above that too must fall
        self.fall(start)

    def fall(self, cell):
        row, col = cell
        while self.surface_array[row + 1][col] == 1:
            self.surface_array[row][col] = 1
            self.surface_array[row + 1][col] = 0
            row += 1

        self.surface_array[row + 1][col] = -1

    def desorption_event(self, args):
        row, col = args[0]

        self.surface_array[row][col] = 0

        # if there are things above they too must fall
        self.fall(args[0])

    def deposition_event(self, args):
        row, col = args[0]
        self.surface_array[row][col] = 1
        if row == len(self.surface_array) - 1:
            new_row = [-1 for _ in range(self.surface_width)]
            self.surface_array.append(new_row)
        self.surface_array[row+1][col] = 0


    # -----------------------------------------------

    def print_grid(self, start=""):
        for row in range(len(self.surface_array)-1, -1, -1):
            print(start, end="")
            for cell in self.surface_array[row]:
                if cell == -1 or cell == 0:
                    print("⬜", end="")
                else:
                    print("🟥", end="")
            print()
        for i in range(self.surface_width):
            print("⬛", end="")
        print()

    def get_normalized_state_rate_string(self, state_rate_list):
        r_string = ""
        running_p_sum = 0

        for action_entry in state_rate_list:
            rate = action_entry[0]
            event = action_entry[1]
            lower_bound = running_p_sum
            upper_bound = running_p_sum + rate
            running_p_sum = upper_bound

            # Format each column to a fixed width
            range_str = f"\t\t[{lower_bound:.6f} - {upper_bound:.6f}]"
            rate_str = f"{rate:.6f}"
            action, args = event

            if action == self.diffusion_event:
                detail_str = f"diffusion  || [{args[0][0]}, {args[0][1]}] ==> [{args[1][0]}, {args[1][1]}]"
            elif action == self.desorption_event:
                detail_str = f"desorption || [{args[0][0]}, {args[0][1]}]"
            else:
                detail_str = f"deposition || [{args[0][0]}, {args[0][1]}]"

            # Fixed-width formatting: adjust padding as needed
            line_string = f"{range_str:<30}||  {rate_str:<10}||  {detail_str}\n"
            r_string += line_string

        return r_string

    def print_hist(self):
        for i in range(len(self.hist)):
            print(f"STEP {i+1}")
            print(f"\tstart_time: {self.hist[i]["start_time"]}")
            print(f"\tstart_time: {self.hist[i]["end_time"]}")
            normalized_state_rate_string = self.get_normalized_state_rate_string(self.hist[i]["normalized_state_rates"])
            print(f"\tnormalized_state_rates:\n{normalized_state_rate_string}")
            print(f"\trandom_num_chosen: {self.hist[i]['random_num_chosen']}")
            print(f"\tAction taken: {self.hist[i]['action_taken'][0]}")
            print(f"\tgrid_after_step:")
            curr_grid = self.hist[i]['grid_after_step']
            for row in range(len(curr_grid) - 1, -1, -1):
                print("\t\t", end="")
                for cell in curr_grid[row]:
                    if cell == -1 or cell == 0:
                        print("⬜", end="")
                    else:
                        print("🟥", end="")
                print()
            print("\t\t", end="")
            for _ in range(self.surface_width):
                print("⬛", end="")
            print()

# Note: Side walls are impenetrable
if __name__ == '__main__':
    s = SimpleSurface(1160, 10000, 10000, 0.70, 0.85, 7, 1)

    s.take_n_steps(20)
    s.print_hist()

