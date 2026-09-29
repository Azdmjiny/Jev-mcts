from jev_mcts.policy import JevPolicy
from jev_mcts.search import MCTS, softmax_weighted_return


class FakeChoices:
    def ask_choice(self, state, instructions, options):
        return {key: (1.0 if value == "good" else 0.0) for key, value in options.items()}


def test_policy_maps_typed_choices_to_actions():
    policy = JevPolicy(FakeChoices())
    assert policy.prior("goal", "scene", [], ["bad", "good"]) == [0.25, 0.75]


def test_softmax_q_favors_successful_rollout():
    value = softmax_weighted_return([0.0, 10.0])
    assert 5.0 < value < 10.0


class OneStepEnv:
    valid_actions = ["bad", "good"]

    def step(self, action):
        self.valid_actions = []
        return "terminal", 10.0 if action == "good" else 0.0, True, []


def test_search_selects_good_action():
    policy = JevPolicy(FakeChoices())
    search = MCTS(policy, simulations=12, max_depth=3, seed=1)
    action, root = search.search(
        "goal", "scene", [], ["bad", "good"], lambda rng: {}, lambda graph: OneStepEnv()
    )
    assert action == "good"
    assert root.visits == 12
