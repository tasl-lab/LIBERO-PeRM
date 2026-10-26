"""Speed / acceleration preferences (motion tier).

Both metrics record an EpisodeTrace and derive the calibration raw from it
at episode end (see episode_trace.py):

  SpeedRangePreference  raw = steps until first success (t_end); the naive
                        ~ pref direction ("less" = speed_up, "more" =
                        slow_down) is encoded by the dual anchors' sign.
  AccRangePreference    raw = P95 of per-step |Δ²p| of the EE over
                        [0, t_end) (m/step²) — robust to the single-step
                        spikes at contact / replan boundaries that dominate
                        the previous peak-based raw.
"""

import numpy as np
from .motion_preference import BaseMotionPreference
from .episode_trace import EpisodeTrace
from typing import Any, Set


def get_gripper_speed(env):
	"""Returns the gripper end-effector linear speed magnitude (m/s)."""
	try:
		eef_name = getattr(env.robots[0], "eef_name", "gripper0_eef")
		v = env.sim.data.get_body_xvelp(eef_name)
		return float(np.linalg.norm(v))
	except Exception:
		return 0.0


def speed_preference():
	"""Legacy band reward factory (kept for gen-side callers)."""

	def _sigmoid(x: float, k: float = 6.0) -> float:
		return 1.0 / (1.0 + float(np.exp(-k * x)))

	def reward(env, params):
		v_low = float(params.get("v_low", params.get("low", 1.0)))
		v_high = float(params.get("v_high", params.get("high", 2.0)))
		if v_high < v_low:
			v_low, v_high = v_high, v_low
		falloff = max(float(params.get("falloff", 0.05)), 1e-6)
		s = get_gripper_speed(env)
		if v_low <= s <= v_high:
			return 1.0
		if s < v_low:
			return float(_sigmoid((s - v_low) / falloff))
		return float(_sigmoid((v_high - s) / falloff))

	return reward


class SpeedRangePreference(BaseMotionPreference):
	"""Pace preference. raw = steps until first success.

	``direction="less"`` (speed_up) / ``"more"`` (slow_down) is informational:
	the dual-anchor satisfaction uses sign(pref_anchor - naive_anchor).
	get_reward returns 1.0 each step (kept for HDF5 per-step logging).
	"""

	saturation = 1.0

	def __init__(
		self,
		min_speed: float = 0.0,
		max_speed: float = float("inf"),
		falloff: float = 0.05,
		direction: str = "less",
		tolerance: float = 0.30,
		pref_description: str = "",
		pos_description: str = "",
	):
		super().__init__(pref_description, pos_description)
		if direction not in {"less", "more"}:
			raise ValueError("direction must be 'less' (speed_up) or 'more' (slow_down)")
		self.min_speed = float(min_speed)
		self.max_speed = float(max_speed)
		self.falloff = float(falloff)
		self.direction = direction
		self.tolerance = float(tolerance)
		self.type = "SpeedRangePreference"
		self.trace = EpisodeTrace()
		# ── per-subtask 计时（2026-09-01）────────────────────────────────
		# task_plan 注入后，用 TaskPlanCompliance 的两阶段判定实时切段：
		# pick = 夹爪接触抓住+抬起，put = 放到目标附近+松爪。raw 从整集步数
		# 变成每个 subtask 的耗时；satisfaction = 各 subtask 的累计双锚
		# （2026-10-03：从"跟 pref 中位数的比例"改成双锚，见 _dual_score）。
		self._subtasks: list | None = None
		self._comp = None                     # TaskPlanCompliance 检测器
		self._done_steps: list[int] = []      # 各 subtask 完成时的 trace 步
		self._cur = 0
		self.subtask_pref_anchor: list | None = None    # calibrate 注入
		self.subtask_naive_anchor: list | None = None

	def set_task_plan(self, task_plan) -> None:
		subs = [(str(s[0]), str(s[1])) for s in (task_plan or [])
		        if isinstance(s, (list, tuple)) and len(s) >= 2 and s[1]]
		self._subtasks = subs or None

	def reset(self, env: Any) -> None:
		self.trace.reset()
		self._done_steps, self._cur, self._comp = [], 0, None
		if self._subtasks:
			try:
				from Preference.high_level.task_plan_progress import (
					TaskPlanCompliancePreference,
				)
				self._comp = TaskPlanCompliancePreference(self._subtasks)
				self._comp.reset(env)
			except Exception:
				self._comp = None

	def mark_success(self) -> None:
		self.trace.mark_success()
		# BDDL 总 goal 达成：没检出的剩余 subtask 在成功步收口
		if self._subtasks:
			while len(self._done_steps) < len(self._subtasks):
				self._done_steps.append(self.trace.t_end)
			self._cur = len(self._subtasks)

	def get_reward(self, env: Any) -> float:
		self.trace.record(env)
		if self._comp is not None and self._cur < len(self._subtasks or []):
			try:
				self._comp.get_reward(env)
				done = self._comp._sub_completed
				# 顺序消费：只有当前 subtask 完成才推进（后面提早 latch 的
				# 会在轮到它时同步收口）
				while self._cur < len(self._subtasks) and done[self._cur]:
					self._done_steps.append(self.trace.n)
					self._cur += 1
			except Exception:
				pass
		return 1.0

	def final_raw(self, mean_reward: float | None = None) -> float | None:
		"""Steps until first success (whole episode if never successful)."""
		if self.trace.n == 0:
			return None
		return float(self.trace.t_end)

	def subtask_raw(self) -> list | None:
		"""每个 subtask 的耗时（步数差分）；未完成的位置为 None。"""
		if not self._subtasks:
			return None
		out, prev = [], 0
		for i in range(len(self._subtasks)):
			if i < len(self._done_steps):
				out.append(float(self._done_steps[i] - prev))
				prev = self._done_steps[i]
			else:
				out.append(None)
		return out

	# 累计锚差至少要到累计 pref 步数的这个比例才出分：抓取那一段带不带 speed
	# 耗时几乎一样（40 步里差 2–3 步，约 7%），单独拿它做分母全是噪声；
	# 搬运段的锚差在 25–30%。
	MIN_ANCHOR_GAP: float = 0.10

	def _has_dual(self) -> bool:
		pa, na = self.subtask_pref_anchor, self.subtask_naive_anchor
		return bool(pa and na and len(pa) == len(na))

	def _dual_score(self, upto: int) -> float | None:
		"""前 upto 个已完成 subtask 的累计双锚分数（0–1），不够区分度时 None。

		score = clip( Σ(R_i − N_i) / Σ(P_i − N_i), 0, 1 )
		P_i / N_i = pref / no-pref demo 在 subtask i 上的耗时中位数。等价于各 subtask
		双锚分数按 |P_i − N_i| 加权平均：speed 真正起作用的搬运段权重大，几乎不受
		影响的抓取段权重小。方向（speed_up / slow_down）由锚差的符号自动带出。"""
		pa, na = self.subtask_pref_anchor, self.subtask_naive_anchor
		sr = self.subtask_raw() or []
		num = den = ptot = 0.0
		n = 0
		for i in range(min(upto, len(sr), len(pa))):
			if sr[i] is None:
				continue
			P, N = float(pa[i]), float(na[i])
			num += sr[i] - N
			den += P - N
			ptot += P
			n += 1
		if n == 0 or abs(den) < max(self.MIN_ANCHOR_GAP * ptot, 1.0):
			return None
		return float(min(max(num / den, 0.0), 1.0))

	def _subtask_score(self, i: int, R: float) -> float | None:
		"""没有 no-pref 锚时的回退（pref-only mission）：跟 pref 中位数比。

		speed_up ("less"): clip(P_i / R, 0, 1)；slow_down ("more"): clip(R / P_i, 0, 1)。
		这个口径下不带 preference 的 rollout 也会落在 50–85，所以只在缺 no-pref 锚时用。"""
		pa = self.subtask_pref_anchor
		if not pa or i >= len(pa):
			return None
		P = float(pa[i])
		if P <= 0 or R <= 0:
			return None
		# yaml 里常不写 direction：speed 倍率 < 1（如 0.5）就是 slow_down
		slow = self.direction == "more" or 0.0 < self.max_speed < 1.0
		ratio = (R / P) if slow else (P / R)
		return float(min(max(ratio, 0.0), 1.0))

	def subtask_points(self) -> list | None:
		"""[(完成步 t_k, 到第 k 段为止的累计分×100)]，供渲染端画
		"每完成一个 subtask 出一个点、点间直线连接" 的曲线。"""
		if not self._subtasks or not self._done_steps:
			return None
		sr = self.subtask_raw() or []
		pts, scores = [], []
		for i, t in enumerate(self._done_steps):
			if i >= len(sr) or sr[i] is None:
				continue
			if self._has_dual():
				s = self._dual_score(i + 1)
				if s is not None:
					pts.append((int(t), s * 100.0))
				continue
			s = self._subtask_score(i, sr[i])
			if s is None:
				continue
			scores.append(s)
			pts.append((int(t), float(np.mean(scores) * 100.0)))
		return pts or None

	def satisfaction(self, mean_reward: float) -> float | None:
		"""双锚（pref 中位数 → 100，no-pref 中位数 → 0），按 subtask 累计。

		优先级：per-subtask 双锚 → 整集步数双锚（基类）→ 只有 pref 锚时的比例回退。"""
		if self._subtasks and self._has_dual():
			s = self._dual_score(len(self._subtasks))
			if s is not None:
				return s * 100.0
			return super().satisfaction(mean_reward)
		if self.subtask_pref_anchor and self._subtasks and self.naive_anchor is None:
			sr = self.subtask_raw() or []
			vals = [self._subtask_score(i, sr[i])
			        for i in range(len(self._subtasks))
			        if i < len(sr) and sr[i] is not None]
			vals = [v for v in vals if v is not None]
			if vals:
				return float(np.mean(vals) * 100.0)
			return None
		return super().satisfaction(mean_reward)

	@property
	def objects_of_interest(self) -> Set[str]:
		return set()


class AccRangePreference(BaseMotionPreference):
	"""Smoothness preference. raw = P95 of |Δ²p_EE| per step over [0, t_end).

	Natural language: "Avoid sudden acceleration" → max_acc low. The per-step
	reward is a band reward on the running P95 (for logging only).
	"""

	saturation = 1.0
	SIGMOID_K: float = 6.0
	PERCENTILE: float = 95.0
	# A P95 over a handful of samples is just "max of a few" — the running
	# curve whipsaws 100 -> 0 in the first ~20 steps before any real signal
	# exists. Report None below this count; the episode-final raw is unaffected
	# (real episodes carry 130+ acceleration samples).
	MIN_SAMPLES: int = 20

	def __init__(
		self,
		min_acc: float = 0.0,
		max_acc: float = float("inf"),
		falloff: float = 0.05,
		pref_description: str = "",
		pos_description: str = "",
	):
		super().__init__(pref_description, pos_description)
		if min_acc < 0:
			raise ValueError("min_acc must be >= 0")
		if max_acc < min_acc:
			raise ValueError("max_acc must be >= min_acc")
		self.min_acc = float(min_acc)
		self.max_acc = float(max_acc)
		self.falloff = max(float(falloff), 1e-6)
		self.type = "AccRangePreference"
		self.trace = EpisodeTrace()

	def reset(self, env: Any) -> None:
		self.trace.reset()

	def mark_success(self) -> None:
		self.trace.mark_success()

	def _band(self, a: float) -> float:
		if self.min_acc <= a <= self.max_acc:
			return 1.0
		s = lambda x: 1.0 / (1.0 + float(np.exp(-self.SIGMOID_K * x)))
		if a < self.min_acc:
			return float(s((a - self.min_acc) / self.falloff))
		return float(s((self.max_acc - a) / self.falloff))

	def get_reward(self, env: Any) -> float:
		self.trace.record(env)
		acc = self.trace.step_accels()
		if len(acc) == 0:
			return 1.0
		return self._band(float(np.percentile(acc, self.PERCENTILE)))

	def final_raw(self, mean_reward: float | None = None) -> float | None:
		acc = self.trace.step_accels()
		if len(acc) < self.MIN_SAMPLES:
			return None
		return float(np.percentile(acc, self.PERCENTILE))

	@property
	def objects_of_interest(self) -> Set[str]:
		return set()
