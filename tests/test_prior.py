from jev_mcts.prior import location_options, object_prior, room_prior


class FakeChoices:
    def ask_choice(self, state, instructions, options):
        assert len(options) >= 2
        return {key: 1 / len(options) for key in options}


def test_location_aliases_are_unique():
    options = location_options(["bathroom_cabinet", "bathroomcabinet"], ["bathroom_counter"])
    assert set(options) == {"INSIDE|bathroomcabinet", "ON|bathroomcounter"}


def test_jev_probabilities_preserved_in_prior():
    prior = object_prior(FakeChoices(), "apple", ["fridge"], ["kitchentable"])
    assert prior == [["INSIDE", "fridge", 0.5], ["ON", "kitchentable", 0.5]]


def test_room_prior_has_four_rooms():
    prior = room_prior(FakeChoices(), "fridge")
    assert len(prior) == 4
    assert sum(row[2] for row in prior) == 1
