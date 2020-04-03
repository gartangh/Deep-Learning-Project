class Agent:
	def __init__(self, color):
		self.color = color
		self.wins = 0
		self.score = 0
		self.num_games_won = 0

	def __str__(self):
		return f'Agent: color={self.color.name}, score={self.score}'

	def next_action(self, legal_actions):
		raise NotImplementedError

	def immediate_reward(self, board, prev_board, turn):
		raise NotImplementedError
