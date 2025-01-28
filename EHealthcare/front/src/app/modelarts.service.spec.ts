import { TestBed } from '@angular/core/testing';

import { ModelartsService } from './modelarts.service';

describe('ModelartsService', () => {
  let service: ModelartsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ModelartsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
