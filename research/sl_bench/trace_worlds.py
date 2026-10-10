"""Fresh controlled families; hidden initial mixtures remain evaluator-only."""
from fractions import Fraction as F
from .oracle import FiniteWorld


def controlled_world(delay=False, noisy=False):
    # Four start states (mode,bit), two holding states, two terminal bits.
    outputs=(0,1,0,1,0,1,0,1)
    kernels=[]
    for action in (0,1):
        rows=[]
        for state in range(8):
            if state<4:
                mode,bit=divmod(state,2)
                target_bit=bit^(mode & action)
                target=(4 if delay else 6)+target_bit
            elif state<6:target=6+(state-4)
            else:target=state
            rows.append(tuple(int(j==target) for j in range(8)))
        kernels.append(tuple(rows))
    # With delay, hide start/holding emissions and reveal only at the third step.
    emission=[]
    for i,bit in enumerate(outputs):
        p=F(1,2) if delay and i<6 else F(bit)
        if noisy and p!=F(1,2):p=F(1,10) if bit==0 else F(9,10)
        emission.append((1-p,p))
    world=FiniteWorld(tuple(kernels),tuple(emission))
    if delay:
        # Same early marginals, opposite delayed states; action controls reversal.
        mixtures=((1,0,0,0,0,0,0,0),(0,0,1,0,0,0,0,0),
                  (0,1,0,0,0,0,0,0),(0,0,0,1,0,0,0,0))
    else:
        mixtures=((F(1,2),F(1,2),0,0,0,0,0,0),(0,0,F(1,2),F(1,2),0,0,0,0),
                  (F(1,2),F(1,2),0,0,0,0,0,0),(0,0,F(1,2),F(1,2),0,0,0,0))
    return world,mixtures


def mixture_profiles(world,mixtures,horizon=3):
    source=world.profiles(horizon)
    return tuple({p:tuple(sum(F(weights[s])*source[s][p][y] for s in range(world.n)) for y in range(len(source[0][p]))) for p in source[0]} for weights in mixtures)


def sample(rng,probabilities):
    u=rng.random();total=0
    for i,p in enumerate(probabilities):
        total+=float(p)
        if u<total:return i
    return len(probabilities)-1


def trial(rng,world,mixture,probe):
    state=sample(rng,mixture);out=[]
    for a in probe:
        out.append(sample(rng,world.emission[state]))
        state=sample(rng,world.kernel[a][state])
    return tuple(out)
