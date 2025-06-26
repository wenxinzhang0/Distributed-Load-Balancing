import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class WorkloadSimulation:
    def __init__(self, epsilon=0.1, initial_N1=10, initial_N2=0):
        """
        Initialize the workload simulation.
        
        Parameters:
        - epsilon: small parameter for arrival rate (lambda = 2 - epsilon)
        - initial_N1: initial workload for backend 1 (should be large)
        - initial_N2: initial workload for backend 2 (should be 0)
        """
        self.epsilon = epsilon
        self.lambda_arrival = 2 - epsilon
        self.initial_N1 = initial_N1
        self.initial_N2 = initial_N2
    
    def service_rate(self, N):
        """
        Service rate function: μ_b(N) = 1 - exp(-N)
        """
        return 1 - np.exp(-N)
    
    def gmsr_routing(self, N1, N2):
        """
        GMSR (Global Minimum Size Routing) policy:
        Route jobs to the backend with smaller workload
        """
        if N1 < N2:
            return 1.0, 0.0  # x1=1, x2=0
        elif N2 < N1:
            return 0.0, 1.0  # x1=0, x2=1
        else:
            return 0.5, 0.5  # equal routing when workloads are equal
    
    def open_loop_routing(self):
        """
        Open-loop routing policy:
        Route jobs equally to both backends
        """
        return 0.5, 0.5  # x1=0.5, x2=0.5
    
    def system_dynamics_gmsr(self, t, y):
        """
        System dynamics under GMSR policy
        dy/dt = [dN1/dt, dN2/dt]
        """
        N1, N2 = y
        
        # Ensure non-negative workloads
        N1 = max(N1, 0)
        N2 = max(N2, 0)
        
        # Get routing proportions
        x1, x2 = self.gmsr_routing(N1, N2)
        
        # Calculate derivatives
        dN1_dt = self.lambda_arrival * x1 - self.service_rate(N1)
        dN2_dt = self.lambda_arrival * x2 - self.service_rate(N2)
        
        return [dN1_dt, dN2_dt]
    
    def system_dynamics_open_loop(self, t, y):
        """
        System dynamics under open-loop routing policy
        dy/dt = [dN1/dt, dN2/dt]
        """
        N1, N2 = y
        
        # Ensure non-negative workloads
        N1 = max(N1, 0)
        N2 = max(N2, 0)
        
        # Get routing proportions
        x1, x2 = self.open_loop_routing()
        
        # Calculate derivatives
        dN1_dt = self.lambda_arrival * x1 - self.service_rate(N1)
        dN2_dt = self.lambda_arrival * x2 - self.service_rate(N2)
        
        return [dN1_dt, dN2_dt]
    
    def simulate(self, t_span=(0, 20), t_eval=None):
        """
        Simulate both policies and return results
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        initial_conditions = [self.initial_N1, self.initial_N2]
        
        # Simulate GMSR policy
        sol_gmsr = solve_ivp(
            self.system_dynamics_gmsr, 
            t_span, 
            initial_conditions, 
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )
        
        # Simulate open-loop policy
        sol_open_loop = solve_ivp(
            self.system_dynamics_open_loop, 
            t_span, 
            initial_conditions, 
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )
        
        return {
            'time': t_eval,
            'gmsr': {
                'N1': sol_gmsr.y[0],
                'N2': sol_gmsr.y[1],
                'success': sol_gmsr.success
            },
            'open_loop': {
                'N1': sol_open_loop.y[0],
                'N2': sol_open_loop.y[1],
                'success': sol_open_loop.success
            }
        }
    
    def plot_results(self, results, save_fig=False, filename='workload_dynamics.png'):
        """
        Plot the simulation results on a single plot comparing both policies
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        time = results['time']
        
        # Calculate the horizontal line value (optimal workload)
        optimal_workload = np.log(2 / self.epsilon)
        
        # Plot both policies on the same plot
        ax.plot(time, results['gmsr']['N1'], 'b-', linewidth=2, label='Backend 1 (GMSR)')
        ax.plot(time, results['gmsr']['N2'], 'r-', linewidth=2, label='Backend 2 (GMSR)')
        ax.plot(time, results['open_loop']['N1'], 'b--', linewidth=2, alpha=0.7, label='Backend 1 (Open-loop)')
        ax.plot(time, results['open_loop']['N2'], 'r--', linewidth=2, alpha=0.7, label='Backend 2 (Open-loop)')
        
        # Add horizontal line for optimal workload
        ax.axhline(y=optimal_workload, color='black', linestyle=':', linewidth=2, alpha=0.8, 
                   label=r'Optimal Workload: $\ln(2/\epsilon)$')
        
        # Title removed as requested
        ax.set_xlabel('Time', fontsize=30)
        ax.set_ylabel('Workload $N_b(t)$', fontsize=30)
        ax.legend(loc='upper right', fontsize=25)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        plt.show()
    
if __name__ == "__main__":
    # Create simulation instance
    sim = WorkloadSimulation(epsilon=0.1, initial_N1=6, initial_N2=0)
    
    print("Starting workload dynamics simulation...")
    print(f"Arrival rate λ = {sim.lambda_arrival}")
    print(f"Service rate function: μ(N) = 1 - exp(-N)")
    print(f"Initial conditions: N1(0) = {sim.initial_N1}, N2(0) = {sim.initial_N2}")
    print()
    
    # Run simulation
    results = sim.simulate(t_span=(0, 50))
    
    # Plot results
    sim.plot_results(results, save_fig=True)
