from dataclasses import dataclass


@dataclass
class Question:
    content: str


@dataclass
class Answer:
    content: str


@dataclass
class SolverRequest:
    content: str
    question: str


@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int


@dataclass
class FinalSolverResponse:
    answer: str
