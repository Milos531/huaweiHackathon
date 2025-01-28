import { TestBed } from '@angular/core/testing';

import { UsercloudService } from './usercloud.service';

describe('UsercloudService', () => {
  let service: UsercloudService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UsercloudService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
