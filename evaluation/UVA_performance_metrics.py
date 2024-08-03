import numpy as np


def output_metrics(metrics, episode):
    print(f"Episode {episode} UAV Performance Metrics:")
    print(f"  Total Distance Traveled: {metrics['total_distance']:.2f}")
    print(f"  Average Distance Per Step: {metrics['average_distance_per_step']:.2f}")
    print(f"  Number of Users Covered: {metrics['num_users_covered']}")
    print(f"  Average Coverage Per Step: {metrics['average_coverage_per_step']:.2f}")
    print(f"  Total Time: {metrics['total_time']}")
    print(f"  Total Rewards: {metrics['total_rewards']:.2f}")
    print(f"  Average Reward Per Step: {metrics['average_reward_per_step']:.2f}")


class UAVPerformanceMetrics:
    def __init__(self):
        self.total_distance = 0
        self.unique_users_covered = set()
        self.total_time = 0
        self.total_rewards = 0
        self.steps = 0
        self.coverage_threshold = 5

    def update(self, uav_positions, user_positions, total_rewards):
        self.steps += 1

        if hasattr(self, 'previous_uav_positions'):
            distances = np.linalg.norm(uav_positions - self.previous_uav_positions, axis=1)
            self.total_distance += np.sum(distances)

        for user_idx, user_pos in enumerate(user_positions):
            if any(np.linalg.norm(user_pos - uav_pos) < self.coverage_threshold for uav_pos in uav_positions):
                self.unique_users_covered.add(user_idx)

        self.total_time += 1
        self.total_rewards = total_rewards

        self.previous_uav_positions = np.copy(uav_positions)

    def get_metrics(self):
        average_distance_per_step = self.total_distance / self.steps if self.steps > 0 else 0
        average_coverage_per_step = len(self.unique_users_covered) / self.steps if self.steps > 0 else 0
        average_reward_per_step = self.total_rewards / self.steps if self.steps > 0 else 0

        return {
            'total_distance': self.total_distance,
            'average_distance_per_step': average_distance_per_step,
            'num_users_covered': len(self.unique_users_covered),
            'average_coverage_per_step': average_coverage_per_step,
            'total_time': self.total_time,
            'total_rewards': self.total_rewards,
            'average_reward_per_step': average_reward_per_step
        }

