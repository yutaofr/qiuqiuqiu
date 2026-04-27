# Warmup Explanation

- Window: 2008_09_to_2009_06 (2008-09-01 to 2009-06-30)
- Blocked: True
- Binding stage: covariance_warmup
- First finite Theta date: 2005-01-28
- Covariance earliest usable date: 2010-01-15
- State-label earliest valid date: 2010-01-22

## Dependency Chain
- raw weekly inputs
- dual-memory factors L_t/T_t/P_t
- Theta(H_t,I_t)
- 260 finite-Theta covariance warmup
- prototype initialization/state assignment
- state_label

## Blocking Reason

The event window ends before state probabilities and semantic labels are unlocked. The binding chain is raw weekly inputs -> dual-memory state factors -> finite Theta(H,I) -> 260 finite-Theta covariance warmup -> prototype initialization/state assignment.
