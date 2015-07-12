JFLAGS = -g
JC = javac
JVM = java
FILE = 
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	NeuralNode.java \
	NeuralNetwork.java \
	Main.java

MAIN = Main

ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

default: classes

classes: $(CLASSES:.java=.class)

run: classes
	$(JVM) $(MAIN) $(RUN_ARGS)

clean:
	$(RM) *.class
