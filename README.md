# Compiler3

This is a playground package to provide companion code to the new compiler
interfaces that I'm working on for base julia. It is organized thusly:

- `exploration/` - Just some scripts for my own reference playing with interfaces
- `dependents/` - Ecosystem packages (imported using git subtree) in the process of being ported to the new interface
- `src/` - Some code that I think would be useful to be generally shared

The idea here is that `exploration` will eventually be deleted, `dependents`
will get merged back to the appropriate upstream package and `src` will become
some sort of Compiler base package (maybe a stdlib, maybe not) that provides
common functionality. I'm hoping by having everything in one repo, iteration
will be quick.
