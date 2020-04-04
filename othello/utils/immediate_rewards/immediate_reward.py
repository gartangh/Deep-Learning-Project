from game_logic.board import Board


class ImmediateReward:
	def immediate_reward(self, board: Board, color_value: int) -> float:
		raise NotImplementedError
