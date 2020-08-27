"""Example robot classifiers.

This module defines a set of minion objects designed to mimic various
classification behaviour.  All minions must implement the minion.classify()  method to determine how they assign classifications to subjects.
"""

import random

class minion(object):
  """Abstract minion class.

  Attributes
  ----------
  id : int
    unique minion id.
  name : str
    unique minion name.

  Methods
  -------
  classify(subject_id)
    classify the given subject.
  """
  def __init__(self, id, name):
    """
    Parameters
    ----------
    id : int
      unique minion id.
    name : str
      unique minion name.
    """
    self.id = id
    self.name = name
  
  def classify(self, subject_id):
    """Abstract class method.
    
    Must be overriden by subclasses.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Raises
    ------
    NotImplementedError
      this class method must be overridden.
    """
    raise NotImplementedError

class ExpertMinion(minion):
  """Expert classifier always returns the correct label.
  
  Classifies a given subject with the expert provided label.

  Attributes
  ----------
  id : int
    unique minion id.
  name : str
    unique minion name.
    
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the provided gold label.
  """
  def __init__(self, id, name):
    """
    Parameters
    ----------
    id : int
      unique minion id.
    name : str
      unique minion name.
    """
    super().__init__(id, name)
  
  def classify(self, subject_id, gold_label):
    """Classify the given subject with the expert provided gold label.
    
    Note
    ----
    The provided gold_label must be valid for the classification task.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
    gold_label : int
      expert provided gold label.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, gold_label)

class AllTheSingleLabelsMinion(minion):
  """Classifier returning a single label only.
  
  Classifies all given subjects with the same label.
  
  Attributes
  ----------
  id : int
    unique minion id.
  name : str
    unique minion name.
  label : int
    label that this classifier will return for all subjects.
      
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the provided gold label.
  """
  def __init__(self, id, name, label):
    """
    Parameters
    ----------
    id : int
      unique minion id.
    name : str
      unique minion name.
    labels : array-like, shape (n_labels,)
      list of valid labels for the classification task.
      
    Note
    ----
    The provided labels must be valid for the classification task.
    """
    super().__init__(id, name)
    self.label = label
  
  def classify(self, subject_id):
    """Classify the given subject with the label for this classifier.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, self.label)

class RandomMinion(minion):
  """Classifier that returns a random label for a given subject.
  
  Classifies each subject with a random label drawn from a provided list of 
  valid labels.

  Attributes
  ----------
  id : int
    unique minion id.
  name : str
    unique minion name.
  labels : array-like, shape (n_labels,)
    list of valid labels for the classification task.
      
  Methods
  -------
  classify(subject_id)
    classify the given subject with the provided gold label.
  """

  def __init__(self, id, name, labels):
    """
    Parameters
    ----------
    id : int
      unique minion id.
    name : str
      unique minion name.
    labels : array-like, shape (n_labels,)
      list of valid labels for the classification task.
      
    Note
    ----
    The provided labels must be valid for the classification task.
    """
    super().__init__(id, name)
    self.labels = labels
  
  def classify(self, subject_id):
    """Classify the given subject with a label selected randomly from the 
    provided labels.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    return (subject_id, random.choice(self.labels))

class NoisyMinion(minion):
  """Classifier returns the correct label a specified fraction of the time.
  
  The provided gold standard label is flipped based on the specified noise for
  this classifier defined in its confusion matrix.
  
  Note
  ----
  The Noisyminion class is currently only implemented for binary classification
  problems.
  
  The Expertminion can be replicated with this class by defining both confusion
  matrix elements to be 1 corresponding to a perfectly astute classifier.
  
  Likewise perfectly obtuse, pessimistic and optimisitic classifiers can be 
  created by setting the corresponding confusion matrix elements to 0 or 1.
  
  Attributes
  ----------
  id : int
    unique minion id.
  name : str
    unique minion name.
  confusion_matrix : array-like, shape (2,)
    array of confusion matrix elements.  First element corresponds to 0 class,
    second element corresponds to 1 class.
      
  Methods
  -------
  classify(subject_id, gold_label)
    classify the given subject with the gold label adding noise based on the 
    confusion matrix.
  """
  def __init__(self, id, name, confusion_matrix):
    """
    Parameters
    ----------
    id : int
      unique minion id.
    name : str
      unique minion name.
    confusion_matrix : array-like, shape (2,)
      array of confusion matrix elements.  First element corresponds to 0 
      class, second element corresponds to 1 class.
    
    Raises
    ------
    ValueError
      if all confusion matrix elements are not in the interval [0,1].
      
    Note
    ----
    The confusion matrix elements must be in the interval [0,1]
    """
    super().__init__(id, name)
    if (confusion_matrix < 0).any() and (confusion_matrix > 1).any():
      raise ValueError('All confusion matrix elements must be in the' \
                    +  'interval [0,1].')
    self.confusion_matrix = confusion_matrix

  def classify(self, subject_id, gold_label):
    """Classify the given subject with the gold label adding noise based on the
    confusion matrix.

    Note
    ----
    The provided gold_label must be valid for the classification task.
    
    Parameters
    ----------
    subject_id : int
      unique id of subject to classify.
    gold_label : int
      expert provided gold label.
      
    Returns
    -------
    subject_id : int
      unique id of classified subject.
    classification : int
      label assigned to the given subject.
    """
    if random.random() < self.confusion_matrix[gold_label]:
      return (subject_id, gold_label)
    else:
      return (subject_id, gold_label == 0)
