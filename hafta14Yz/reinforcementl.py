"""Pygame ile yol cizme ve Q-learning tabanli arac surme demosu.

Kullanim:
1. Sol fare tusuyla yol cizin.
2. Enter ile yolu onaylayin.
3. Space ile egitimi baslatip duraklatin.
4. R ile ajanlari sifirlayin, C ile yolu temizleyin.
5. 1 ile hazir demo yolunu yukleyin.

Bu ornek, birden fazla kucuk aracin ayni yol uzerinde paylastigi bir
Q-learning policy'si ile nasil hizli sekilde ilerleyebilecegini gosterir.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

try:
	import pygame
except ImportError as exc:  # pragma: no cover - import guard
	raise SystemExit("This demo requires pygame. Install it with: uv pip install pygame") from exc


WIDTH = 1200
HEIGHT = 700
FPS = 60

BG_TOP = (14, 18, 28)
BG_BOTTOM = (28, 36, 58)
PANEL = (18, 24, 38)
PANEL_EDGE = (78, 90, 120)
TEXT = (245, 247, 255)
MUTED = (168, 176, 202)
GREEN = (93, 224, 154)
RED = (255, 110, 122)
AMBER = (255, 197, 87)
BLUE = (86, 165, 255)
CYAN = (113, 214, 255)
WHITE = (250, 250, 252)
ROAD = (44, 51, 72)
ROAD_EDGE = (116, 126, 160)

DRAWING_WIDTH = 44
TRACK_WIDTH = 58
CAR_COUNT = 12
MAX_STEPS_PER_EPISODE = 780
TRAJECTORY_HISTORY = 220

STEERING_ACTIONS = (-12.0, -6.0, 0.0, 6.0, 12.0)


def clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
	return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
	while angle <= -math.pi:
		angle += math.tau
	while angle > math.pi:
		angle -= math.tau
	return angle


def lerp_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
	amount = clamp(amount, 0.0, 1.0)
	return tuple(int(color_a[index] + (color_b[index] - color_a[index]) * amount) for index in range(3))


def make_gradient_background(size: tuple[int, int]) -> pygame.Surface:
	surface = pygame.Surface(size)
	height = size[1]
	for y in range(height):
		t = y / max(1, height - 1)
		pygame.draw.line(surface, lerp_color(BG_TOP, BG_BOTTOM, t), (0, y), (size[0], y))
	return surface


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
	return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_to_vector(angle: float) -> tuple[float, float]:
	return math.cos(angle), math.sin(angle)


def perpendicular(vector: tuple[float, float]) -> tuple[float, float]:
	return -vector[1], vector[0]


def resample_polyline(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
	if len(points) < 2:
		return points[:]

	resampled = [points[0]]
	carry = 0.0
	previous = points[0]

	for current in points[1:]:
		segment_length = distance(previous, current)
		if segment_length == 0:
			continue

		direction = ((current[0] - previous[0]) / segment_length, (current[1] - previous[1]) / segment_length)
		while carry + segment_length >= spacing:
			remain = spacing - carry
			next_point = (previous[0] + direction[0] * remain, previous[1] + direction[1] * remain)
			resampled.append(next_point)
			previous = next_point
			segment_length -= remain
			carry = 0.0
		carry += segment_length
		previous = current

	if distance(resampled[-1], points[-1]) > 1.0:
		resampled.append(points[-1])

	return resampled


def smooth_polyline(points: list[tuple[float, float]], passes: int = 2) -> list[tuple[float, float]]:
	if len(points) < 3:
		return points[:]

	smoothed = points[:]
	for _ in range(passes):
		next_points = [smoothed[0]]
		for index in range(1, len(smoothed) - 1):
			prev_point = smoothed[index - 1]
			point = smoothed[index]
			next_point = smoothed[index + 1]
			blended = (
				point[0] * 0.5 + (prev_point[0] + next_point[0]) * 0.25,
				point[1] * 0.5 + (prev_point[1] + next_point[1]) * 0.25,
			)
			next_points.append(blended)
		next_points.append(smoothed[-1])
		smoothed = next_points
	return smoothed


def draw_rounded_polyline(
	surface: pygame.Surface,
	points: list[tuple[float, float]],
	color: tuple[int, int, int],
	thickness: int,
) -> None:
	if len(points) < 2:
		return

	radius = max(1, thickness // 2)
	for start, end in zip(points, points[1:]):
		pygame.draw.line(surface, color, start, end, thickness)
	for point in points:
		pygame.draw.circle(surface, color, point, radius)


def build_demo_track() -> list[tuple[float, float]]:
	points: list[tuple[float, float]] = []
	for step in range(88):
		t = step / 87.0
		x = 120 + t * 1120
		y = 420 + math.sin(t * math.tau * 1.1) * 150 + math.sin(t * math.tau * 3.0) * 45
		points.append((x, y))
	return points


@dataclass
class Track:
	points: list[tuple[float, float]]
	width: int = TRACK_WIDTH
	centerline: list[tuple[float, float]] = field(default_factory=list)
	tangents: list[tuple[float, float]] = field(default_factory=list)
	normals: list[tuple[float, float]] = field(default_factory=list)
	cumulative: list[float] = field(default_factory=list)

	@classmethod
	def from_points(cls, points: list[tuple[float, float]], width: int = TRACK_WIDTH) -> Track:
		smoothed = smooth_polyline(points, passes=2)
		centerline = resample_polyline(smoothed, spacing=10.0)

		if len(centerline) < 2:
			raise ValueError("A track needs at least two usable points.")

		tangents: list[tuple[float, float]] = []
		normals: list[tuple[float, float]] = []
		cumulative: list[float] = [0.0]

		for index, point in enumerate(centerline):
			if index == len(centerline) - 1:
				tangent = tangents[-1] if tangents else (1.0, 0.0)
			else:
				next_point = centerline[index + 1]
				dx = next_point[0] - point[0]
				dy = next_point[1] - point[1]
				length = math.hypot(dx, dy) or 1.0
				tangent = (dx / length, dy / length)
			normal = perpendicular(tangent)
			tangents.append(tangent)
			normals.append(normal)
			if index > 0:
				cumulative.append(cumulative[-1] + distance(centerline[index - 1], point))

		return cls(points=points[:], width=width, centerline=centerline, tangents=tangents, normals=normals, cumulative=cumulative)

	@property
	def length(self) -> float:
		return self.cumulative[-1] if self.cumulative else 0.0

	@property
	def start_position(self) -> tuple[float, float]:
		return self.centerline[0]

	@property
	def start_angle(self) -> float:
		tangent = self.tangents[0]
		return math.atan2(tangent[1], tangent[0])

	def nearest_index(self, position: tuple[float, float], hint: int = 0) -> int:
		if len(self.centerline) == 1:
			return 0

		start = clamp_int(hint - 24, 0, len(self.centerline) - 1)
		end = clamp_int(hint + 36, 0, len(self.centerline) - 1)
		best_index = start
		best_distance = float("inf")
		for index in range(start, end + 1):
			candidate_distance = distance(position, self.centerline[index])
			if candidate_distance < best_distance:
				best_distance = candidate_distance
				best_index = index
		if best_distance < float("inf"):
			return best_index

		best_index = 0
		best_distance = float("inf")
		for index, point in enumerate(self.centerline):
			candidate_distance = distance(position, point)
			if candidate_distance < best_distance:
				best_distance = candidate_distance
				best_index = index
		return best_index

	def pose_data(self, position: tuple[float, float], hint: int = 0) -> tuple[int, tuple[float, float], tuple[float, float], float, float]:
		index = self.nearest_index(position, hint)
		center = self.centerline[index]
		tangent = self.tangents[index]
		normal = self.normals[index]
		offset_vector = (position[0] - center[0], position[1] - center[1])
		lateral = offset_vector[0] * normal[0] + offset_vector[1] * normal[1]
		curve = 0.0
		if index < len(self.tangents) - 1:
			next_tangent = self.tangents[index + 1]
			curve = normalize_angle(math.atan2(next_tangent[1], next_tangent[0]) - math.atan2(tangent[1], tangent[0]))
		return index, center, tangent, lateral, curve

	def draw(self, screen: pygame.Surface) -> None:
		if len(self.centerline) < 2:
			return

		outer_points = []
		inner_points = []
		half_width = self.width * 0.5
		for point, normal in zip(self.centerline, self.normals, strict=False):
			outer_points.append((point[0] + normal[0] * half_width, point[1] + normal[1] * half_width))
			inner_points.append((point[0] - normal[0] * half_width, point[1] - normal[1] * half_width))

		draw_rounded_polyline(screen, outer_points, ROAD_EDGE, 4)
		draw_rounded_polyline(screen, inner_points, ROAD_EDGE, 4)
		draw_rounded_polyline(screen, self.centerline, ROAD, self.width)
		draw_rounded_polyline(screen, self.centerline, (72, 82, 108), max(2, self.width // 10))

		for index in range(0, len(self.centerline), 5):
			point = self.centerline[index]
			normal = self.normals[index]
			lane_width = self.width * 0.2
			a = (point[0] - normal[0] * lane_width, point[1] - normal[1] * lane_width)
			b = (point[0] + normal[0] * lane_width, point[1] + normal[1] * lane_width)
			pygame.draw.line(screen, (150, 160, 192), a, b, 2)

		pygame.draw.circle(screen, GREEN, self.centerline[0], 7)
		pygame.draw.circle(screen, RED, self.centerline[-1], 7)


@dataclass
class QLearningAgent:
	action_count: int = len(STEERING_ACTIONS)
	alpha: float = 0.18
	gamma: float = 0.94
	epsilon: float = 0.95
	epsilon_min: float = 0.04
	epsilon_decay: float = 0.992
	q_table: dict[tuple[int, ...], list[float]] = field(default_factory=dict)

	def reset(self) -> None:
		self.q_table.clear()
		self.epsilon = 0.95

	def values(self, state: tuple[int, ...]) -> list[float]:
		if state not in self.q_table:
			self.q_table[state] = [0.0] * self.action_count
		return self.q_table[state]

	def choose_action(self, state: tuple[int, ...], explore: bool = True) -> int:
		if explore and random.random() < self.epsilon:
			return random.randrange(self.action_count)
		values = self.values(state)
		best_value = max(values)
		best_actions = [index for index, value in enumerate(values) if value == best_value]
		return random.choice(best_actions)

	def update(self, state: tuple[int, ...], action: int, reward: float, next_state: tuple[int, ...], done: bool) -> None:
		current_values = self.values(state)
		next_values = self.values(next_state)
		target = reward if done else reward + self.gamma * max(next_values)
		current_values[action] += self.alpha * (target - current_values[action])

	def finish_episode(self) -> None:
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


@dataclass
class Car:
	color: tuple[int, int, int]
	x: float = 0.0
	y: float = 0.0
	angle: float = 0.0
	speed: float = 3.25
	alive: bool = True
	steps: int = 0
	total_reward: float = 0.0
	last_index: int = 0
	trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=36))

	def reset(self, track: Track, offset: float = 0.0) -> None:
		start = track.start_position
		tangent = track.tangents[0]
		normal = track.normals[0]
		self.x = start[0] + normal[0] * offset
		self.y = start[1] + normal[1] * offset
		self.angle = math.atan2(tangent[1], tangent[0])
		self.alive = True
		self.steps = 0
		self.total_reward = 0.0
		self.last_index = 0
		self.trail.clear()
		self.trail.append((self.x, self.y))

	def current_state(self, track: Track) -> tuple[int, int, int, int]:
		index, _, tangent, lateral, curve = track.pose_data((self.x, self.y), self.last_index)
		progress_bucket = clamp_int(int(index / max(1, len(track.centerline) - 1) * 9), 0, 9)
		lateral_bucket = clamp_int(int(lateral / max(1.0, track.width * 0.22)), -4, 4)
		heading_error = normalize_angle(self.angle - math.atan2(tangent[1], tangent[0]))
		heading_bucket = clamp_int(int(heading_error / (math.pi / 12)), -4, 4)
		curve_bucket = clamp_int(int(curve / (math.pi / 14)), -4, 4)
		return progress_bucket, lateral_bucket, heading_bucket, curve_bucket

	def step(self, action_index: int, track: Track) -> tuple[float, bool, tuple[int, int, int, int]]:
		if not self.alive:
			return 0.0, True, self.current_state(track)

		previous_position = (self.x, self.y)
		self.angle = normalize_angle(self.angle + math.radians(STEERING_ACTIONS[action_index]))
		direction = angle_to_vector(self.angle)
		self.x += direction[0] * self.speed
		self.y += direction[1] * self.speed
		self.steps += 1
		self.trail.append((self.x, self.y))

		index, center, tangent, lateral, _ = track.pose_data((self.x, self.y), self.last_index)
		self.last_index = index

		movement = (self.x - previous_position[0], self.y - previous_position[1])
		forward = movement[0] * tangent[0] + movement[1] * tangent[1]
		heading_error = normalize_angle(self.angle - math.atan2(tangent[1], tangent[0]))
		center_distance = distance((self.x, self.y), center)

		reward = forward * 0.58
		reward -= abs(lateral) / max(1.0, track.width) * 0.14
		reward -= abs(heading_error) * 0.12

		done = False
		road_limit = max(10.0, track.width * 0.5 - 6.0)
		if abs(lateral) > road_limit or center_distance > road_limit * 1.15:
			reward -= 4.5
			done = True
		elif index >= len(track.centerline) - 2:
			reward += 6.0
			done = True

		if self.steps >= MAX_STEPS_PER_EPISODE:
			reward -= 1.0
			done = True

		if done:
			self.alive = False

		self.total_reward += reward
		return reward, done, self.current_state(track)

	def draw(self, screen: pygame.Surface) -> None:
		if len(self.trail) > 1:
			pygame.draw.lines(screen, lerp_color(self.color, WHITE, 0.35), False, list(self.trail), 2)

		heading = angle_to_vector(self.angle)
		normal = perpendicular(heading)
		front = (self.x + heading[0] * 12, self.y + heading[1] * 12)
		back_left = (self.x - heading[0] * 9 + normal[0] * 5, self.y - heading[1] * 9 + normal[1] * 5)
		back_right = (self.x - heading[0] * 9 - normal[0] * 5, self.y - heading[1] * 9 - normal[1] * 5)
		pygame.draw.polygon(screen, self.color, [front, back_left, back_right])
		pygame.draw.polygon(screen, WHITE, [front, back_left, back_right], 1)


@dataclass
class EpisodeStats:
	episode: int = 0
	completed: int = 0
	average_reward: float = 0.0
	best_reward: float = -9999.0
	epsilon: float = 0.95


class RoadRacerDemo:
	def __init__(self) -> None:
		pygame.init()
		display_info = pygame.display.Info()
		global WIDTH, HEIGHT
		WIDTH = min(1500, max(1280, display_info.current_w - 80))
		HEIGHT = min(900, max(800, display_info.current_h - 80))
		pygame.display.set_caption("Road Racer RL")
		self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
		self.clock = pygame.time.Clock()
		self.title_font = pygame.font.Font(None, 30)
		self.small_font = pygame.font.Font(None, 22)
		self.background = make_gradient_background((WIDTH, HEIGHT))

		self.user_drawing = True
		self.drawing_points: list[tuple[float, float]] = []
		self.track: Track | None = None
		self.agent = QLearningAgent()
		self.cars: list[Car] = []
		self.stats = EpisodeStats()
		self.reward_history: deque[float] = deque(maxlen=TRAJECTORY_HISTORY)
		self.best_episode_reward = -9999.0
		self.episode_step = 0
		self.paused = False
		self.message = "Sol tus ile yol cizin, Enter ile baslatin."
		self.sample_track = build_demo_track()

		left_column_width = max(320, min(360, WIDTH // 4))
		gap = 18
		track_x = left_column_width + gap * 2
		track_width = max(640, WIDTH - track_x - gap)
		panel_left = gap
		panel_top = gap
		self.control_rect = pygame.Rect(panel_left, panel_top, left_column_width, 320)
		self.stats_rect = pygame.Rect(panel_left, self.control_rect.bottom + gap, left_column_width, 220)
		self.chart_rect = pygame.Rect(panel_left, self.stats_rect.bottom + gap, left_column_width, max(180, HEIGHT - self.stats_rect.bottom - gap * 2))
		self.track_rect = pygame.Rect(track_x, gap, track_width, HEIGHT - gap * 2)

	def reset_agent(self) -> None:
		self.agent.reset()
		if self.track is not None:
			self.start_episode()

	def load_demo_track(self) -> None:
		self.drawing_points = self.sample_track[:]
		self.finish_track()

	def clear_track(self) -> None:
		self.track = None
		self.user_drawing = True
		self.drawing_points.clear()
		self.cars.clear()
		self.reward_history.clear()
		self.stats = EpisodeStats()
		self.episode_step = 0
		self.message = "Yol temizlendi. Sol tus ile yeniden cizebilirsiniz."

	def finish_track(self) -> None:
		if len(self.drawing_points) < 6:
			self.message = "Yol icin daha fazla nokta cizin."
			return

		try:
			self.track = Track.from_points(self.drawing_points, width=TRACK_WIDTH)
		except ValueError:
			self.message = "Bu cizimden kullanilabilir yol olusmadi."
			return

		self.user_drawing = False
		self.message = "Yol hazir. Space ile egitimi baslatin."
		self.agent.reset()
		self.start_episode()

	def start_episode(self) -> None:
		if self.track is None:
			return

		self.cars = []
		for index in range(CAR_COUNT):
			hue = index / max(1, CAR_COUNT - 1)
			color = lerp_color((95, 190, 255), (255, 148, 98), hue)
			car = Car(color=color)
			offset = (index - (CAR_COUNT - 1) / 2.0) * 3.2
			car.reset(self.track, offset=offset)
			self.cars.append(car)

		self.episode_step = 0
		self.stats.episode += 1
		self.stats.completed = 0
		self.message = f"Episode {self.stats.episode} basladi."

	def maybe_restart_episode(self) -> None:
		if self.track is None:
			return
		if all(not car.alive for car in self.cars) or self.episode_step >= MAX_STEPS_PER_EPISODE:
			episode_reward = sum(car.total_reward for car in self.cars) / max(1, len(self.cars))
			self.reward_history.append(episode_reward)
			self.best_episode_reward = max(self.best_episode_reward, episode_reward)
			self.stats.average_reward = episode_reward
			self.stats.best_reward = max(self.stats.best_reward, episode_reward)
			self.stats.epsilon = self.agent.epsilon
			self.agent.finish_episode()
			self.start_episode()

	def update(self) -> None:
		if self.track is None or self.paused:
			return

		self.episode_step += 1
		completed_now = 0

		for car in self.cars:
			if not car.alive:
				continue
			state = car.current_state(self.track)
			action = self.agent.choose_action(state, explore=True)
			reward, done, next_state = car.step(action, self.track)
			self.agent.update(state, action, reward, next_state, done)
			if done:
				completed_now += 1

		self.stats.completed = sum(1 for car in self.cars if not car.alive)
		self.stats.epsilon = self.agent.epsilon
		if completed_now > 0:
			self.message = f"{completed_now} arac hedefe veya sinire ulasti."

		self.maybe_restart_episode()

	def draw_background(self) -> None:
		self.screen.blit(self.background, (0, 0))
		accent = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
		for index in range(9):
			alpha = 16 - index
			radius = 160 + index * 42
			pygame.draw.circle(accent, (255, 255, 255, max(0, alpha)), (WIDTH - 160, 130), radius, width=2)
		self.screen.blit(accent, (0, 0))

	def draw_panel(self, rect: pygame.Rect) -> None:
		pygame.draw.rect(self.screen, PANEL, rect, border_radius=20)
		pygame.draw.rect(self.screen, PANEL_EDGE, rect, width=2, border_radius=20)

	def draw_track_area(self) -> None:
		self.draw_panel(self.track_rect)
		if self.track is not None:
			self.track.draw(self.screen)
			for car in self.cars:
				car.draw(self.screen)
		else:
			if len(self.drawing_points) > 1:
				draw_rounded_polyline(self.screen, self.drawing_points, CYAN, DRAWING_WIDTH)
				draw_rounded_polyline(self.screen, self.drawing_points, WHITE, 2)

		title = self.title_font.render("Road Drawing + Reinforcement Learning", True, TEXT)
		subtitle = self.small_font.render("Cizim modu: fare ile yol cizin | Enter: onayla | Space: egitim", True, MUTED)
		self.screen.blit(title, (self.track_rect.left + 22, self.track_rect.top + 16))
		self.screen.blit(subtitle, (self.track_rect.left + 22, self.track_rect.top + 47))

		if self.track is None:
			info = self.small_font.render("Henuz yol yok. Baslamak icin sol tusu basili tutun.", True, AMBER)
			self.screen.blit(info, (self.track_rect.centerx - info.get_width() // 2, self.track_rect.centery - 12))

	def draw_controls(self) -> None:
		self.draw_panel(self.control_rect)
		title = self.title_font.render("Kontroller", True, TEXT)
		self.screen.blit(title, (self.control_rect.left + 20, self.control_rect.top + 16))

		controls = [
			"Sol tus: yol ciz",
			"Enter: yolu onayla",
			"Space: pause / resume",
			"C: yolu temizle",
			"1: demo yol yukle",
			"R: Q-table sifirla",
			"Esc: cikis",
		]
		start_y = self.control_rect.top + 58
		for index, line in enumerate(controls):
			text = self.small_font.render(line, True, TEXT if index < 3 else MUTED)
			self.screen.blit(text, (self.control_rect.left + 16, start_y + index * 28))

		status = [
			f"Mode: {'Drawing' if self.track is None else 'Training'}",
			f"Episode: {self.stats.episode}",
			f"Epsilon: {self.agent.epsilon:.3f}",
			f"Cars alive: {sum(1 for car in self.cars if car.alive)}",
		]
		status_y = self.control_rect.bottom - 108
		for index, line in enumerate(status):
			color = GREEN if index == 0 and self.track is not None else TEXT
			text = self.small_font.render(line, True, color if index == 0 else MUTED)
			self.screen.blit(text, (self.control_rect.left + 16, status_y + index * 24))

	def draw_stats(self) -> None:
		self.draw_panel(self.stats_rect)
		title = self.title_font.render("Episode Ozeti", True, TEXT)
		self.screen.blit(title, (self.stats_rect.left + 20, self.stats_rect.top + 16))

		if self.reward_history:
			current_reward = self.reward_history[-1]
		else:
			current_reward = 0.0

		values = [
			("Son episode odulu", f"{current_reward:.2f}"),
			("En iyi episode", f"{self.best_episode_reward:.2f}" if self.best_episode_reward > -9990 else "-"),
			("Gecen adim", f"{self.episode_step}"),
			("Basari", f"{self.stats.completed}/{CAR_COUNT}"),
		]

		for index, (label, value) in enumerate(values):
			y = self.stats_rect.top + 66 + index * 32
			self.screen.blit(self.small_font.render(label, True, MUTED), (self.stats_rect.left + 20, y))
			value_color = GREEN if index in (1, 3) else AMBER
			value_surface = self.small_font.render(value, True, value_color)
			self.screen.blit(value_surface, (self.stats_rect.right - 20 - value_surface.get_width(), y))

		footer = self.small_font.render(self.message, True, TEXT)
		self.screen.blit(footer, (self.stats_rect.left + 16, self.stats_rect.bottom - 30))

	def draw_chart(self) -> None:
		self.draw_panel(self.chart_rect)
		title = self.title_font.render("Ogrenme Grafigi", True, TEXT)
		self.screen.blit(title, (self.chart_rect.left + 20, self.chart_rect.top + 16))

		chart = pygame.Rect(self.chart_rect.left + 22, self.chart_rect.top + 56, self.chart_rect.width - 44, self.chart_rect.height - 74)
		pygame.draw.rect(self.screen, (22, 28, 42), chart, border_radius=18)

		if len(self.reward_history) < 2:
			hint = self.small_font.render("Episode odulleri burada gorunecek.", True, MUTED)
			self.screen.blit(hint, hint.get_rect(center=chart.center))
			return

		min_reward = min(self.reward_history)
		max_reward = max(self.reward_history)
		if abs(max_reward - min_reward) < 1e-6:
			max_reward += 1.0
			min_reward -= 1.0

		def point(index: int, value: float) -> tuple[int, int]:
			x = chart.left + int(index / max(1, len(self.reward_history) - 1) * chart.width)
			normalized = (value - min_reward) / (max_reward - min_reward)
			y = chart.bottom - int(normalized * chart.height)
			return x, y

		points = [point(index, value) for index, value in enumerate(self.reward_history)]
		pygame.draw.lines(self.screen, GREEN, False, points, 3)
		for index in range(5):
			y = chart.top + int(index / 4 * chart.height)
			pygame.draw.line(self.screen, (41, 49, 68), (chart.left, y), (chart.right, y), 1)

		current = self.small_font.render(f"Son: {self.reward_history[-1]:.2f}", True, GREEN)
		best = self.small_font.render(f"En iyi: {self.best_episode_reward:.2f}", True, AMBER)
		self.screen.blit(current, (chart.left + 8, chart.top + 8))
		self.screen.blit(best, (chart.right - best.get_width() - 8, chart.top + 8))

	def handle_event(self, event: pygame.event.Event) -> bool:
		if event.type == pygame.QUIT:
			return False

		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				return False
			if event.key == pygame.K_SPACE:
				if self.track is not None:
					self.paused = not self.paused
					self.message = "Egitim duraklatildi." if self.paused else "Egitim devam ediyor."
			elif event.key == pygame.K_RETURN:
				if self.user_drawing:
					self.finish_track()
			elif event.key == pygame.K_c:
				self.clear_track()
			elif event.key == pygame.K_r:
				self.reset_agent()
				self.message = "Q-table sifirlandi."
			elif event.key == pygame.K_1:
				self.load_demo_track()

		if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.track is None:
			self.user_drawing = True
			self.drawing_points = [event.pos]

		if event.type == pygame.MOUSEMOTION and self.user_drawing and self.track is None:
			if event.buttons[0]:
				if not self.drawing_points or distance(self.drawing_points[-1], event.pos) > 7.0:
					self.drawing_points.append(event.pos)

		if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.user_drawing and self.track is None:
			if not self.drawing_points or distance(self.drawing_points[-1], event.pos) > 1.0:
				self.drawing_points.append(event.pos)

		return True

	def run(self) -> None:
		running = True
		print("Sol tus ile yol cizin, Enter ile onaylayin, Space ile egitimi baslatin/durdurun.")
		while running:
			self.clock.tick(FPS)
			for event in pygame.event.get():
				running = self.handle_event(event)

			self.update()
			self.draw_background()
			self.draw_track_area()
			self.draw_controls()
			self.draw_stats()
			self.draw_chart()

			footer = self.small_font.render(
				f"Mode: {'Paused' if self.paused else 'Running'}   |   Q-table size: {len(self.agent.q_table)}   |   Episode {self.stats.episode}",
				True,
				WHITE,
			)
			self.screen.blit(footer, (24, HEIGHT - 28))

			pygame.display.flip()

		pygame.quit()


def main() -> None:
	demo = RoadRacerDemo()
	demo.run()


if __name__ == "__main__":
	main()
