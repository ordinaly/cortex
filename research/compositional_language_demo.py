"""Deterministic abstraction, transfer, conflict and revision demonstration."""
from dataclasses import asdict
import json

from compositional_language import (Symbol, Entity, Atom, Fact, LanguageModel,
    MotifExample, RuleExample, invent_motif, propose_rule, validate_rule)


def main():
    r = Symbol('R7', 1, ('entity', 'entity'))
    s = Symbol('R2', 1, ('entity', 'entity'))
    target = Symbol('R9', 1, ('entity', 'entity'))
    def atom(symbol, a, b, polarity=1):
        return Atom(symbol, (Entity(a), Entity(b)), polarity)
    def example(prefix, polarity=1):
        return RuleExample(prefix, (atom(r, prefix+'a', prefix+'b'), atom(s, prefix+'b', prefix+'c')),
                           atom(target, prefix+'a', prefix+'c', polarity))
    discovery = [example('d1'), example('d2')]
    definition = invent_motif([MotifExample(e.evidence_id, e.premises) for e in discovery])
    rule = propose_rule(discovery)
    validation = validate_rule(rule, [example('v1'), example('v2')])
    if not validation.accepted:
        raise RuntimeError('rule did not pass fresh validation')
    unseen = example('new')
    model = LanguageModel([Fact('obs0','episode',unseen.premises[0]),
                           Fact('obs1','episode',unseen.premises[1])], [definition, rule])
    explanation = model.explain(unseen.observed, 'episode')
    dependencies = sorted({f for n in explanation.nodes for f in n.fact_dependencies})
    model.retract_fact('obs1')
    retracted = model.explain(unseen.observed, 'episode').reason
    model.restore_fact('obs1')
    restored = model.explain(unseen.observed, 'episode').reason
    model.set_rule_active(rule.rule_id, False)
    withdrawn = model.explain(unseen.observed, 'episode').reason
    print(json.dumps({'invented_symbol': definition.head.symbol.name,
                      'exposed_roles': len(definition.head.terms), 'rule': rule.rule_id,
                      'validation': asdict(validation), 'unseen_inference': explanation.reason,
                      'fact_dependencies': dependencies, 'after_evidence_retraction': retracted,
                      'after_evidence_restoration': restored, 'after_rule_withdrawal': withdrawn,
                      'counterexample_gate': asdict(validate_rule(rule, [example('v3'),example('counter',-1)]))}, indent=2))


if __name__ == '__main__':
    main()
