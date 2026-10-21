"""Run the ambiguity-preserving cup fixture with the finite research oracle."""
from fractions import Fraction as F
from possibility_geometry import Candidate, FiniteModel, PossibilityFilter


def main():
    rows = {}
    for candidate, outcome, floor in [('A', 'broken', 'hard'),
                                      ('B', 'intact', 'hard'),
                                      ('C', 'intact', 'soft')]:
        rows[candidate, 'drop'] = {(candidate, outcome): 1}
        rows[candidate, 'inspect'] = {(candidate, floor): 1}
    model = FiniteModel('cup-v1', tuple(Candidate(v, ('cup-memory',)) for v in 'ABC'),
                        ('drop', 'inspect'), ('broken', 'intact', 'hard', 'soft'), rows)
    learner = PossibilityFilter(model, dict.fromkeys('ABC', F(1, 3)))
    for action, observation in [('drop', 'intact'), ('inspect', 'hard')]:
        forecast = learner.predict(action)
        print('forecast', action, {o: str(p) for o, p in forecast.outcomes.items()})
        result = learner.assimilate(action, observation)
        print('observed', observation, result.status,
              {v: str(p) for v, p in result.posterior.items()})


if __name__ == '__main__':
    main()
