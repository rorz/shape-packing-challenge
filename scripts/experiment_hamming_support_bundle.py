"""Isolated support-bundle runner with exact Hamming-ordered interval relations."""
from __future__ import annotations
import sys
import experiment_support_bundle as bundle
import interval_pair_relation as relations
from interval_order_optimization import add_hamming_interval_pair


def main():
 # The support-bundle runner imports this builder at call time. Change the
 # attribute only inside this process, and always restore it for import users.
 original=relations.add_allowed_interval_pair
 try:
  relations.add_allowed_interval_pair=add_hamming_interval_pair
  if '--intervals'not in sys.argv:sys.argv.append('--intervals')
  bundle.main()
 finally:relations.add_allowed_interval_pair=original


if __name__=='__main__':main()
