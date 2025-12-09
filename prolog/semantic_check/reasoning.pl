living(john).
living(mary).
edible(apple).
edible(sandwich).
vehicle(car).

can_act(Actor,eats,Target):-living(Actor),edible(Target).
can_act(Actor,drives,Target):-living(Actor),vehicle(Target).