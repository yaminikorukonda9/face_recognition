import face_recognition
import numpy as np
from sklearn.neighbors import KDTree

import constants


def _linear_search(known_encodings, query_encodings, known_names):
    names = []
    # loop over the facial embeddings
    for encoding in query_encodings:
       
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = constants.ID_UNKNOWN

        # check to see if we have found a match
        if True in matches:
            
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = known_names[i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes 
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    return names


def _find_best_match_within_tolerance(candidates, names, tolerance):
    zipped_dist_names = np.dstack(candidates)
    best_candidates = []

    for candidates in zipped_dist_names:
        count = {}
        best_candidate = constants.ID_UNKNOWN
        filtered_candidates = [int(ind) for dist, ind in candidates if dist <= tolerance]

        if len(filtered_candidates) != 0:
            for ind in filtered_candidates:
                count[names[ind]] = count.get(names[ind], 0) + 1
            best_candidate = max(count, key=count.get)

        best_candidates.append(best_candidate)
    return best_candidates


# Determine the best match for the query encodings within a specified tolerance, returning matched actor names or constants.ID_UNKNOWN if no valid match is found.
# find the k nearest neighbors using precomputed kdtree of training encodings
def _fast_face_match_knn(known_encodings, query_encodings, known_names, tolerance, k):
    kdtree = known_encodings
    results = kdtree.query(query_encodings, k)
    return _find_best_match_within_tolerance(results, known_names, tolerance)


class FaceRec:

    @staticmethod
    def getAllFacesInImage(image_rgb, detection_method, use_fastnn, known_encodings,
                           known_encodings_structure, known_names):

        # Detect bounding box coordinates for each face in the input image and compute facial embeddings for each face.
        boxes = face_recognition.face_locations(image_rgb, model=detection_method)
        encodings = face_recognition.face_encodings(image_rgb, boxes)

        names = []

        if encodings:
            if use_fastnn or known_encodings_structure == constants.ENCD_KDTREE:
                # check if kdtree is to be recomputed or not
                if known_encodings_structure != constants.ENCD_KDTREE:
                    known_encodings = KDTree(np.asarray(encodings), leaf_size=constants.LEAF_SIZE_KDTREE)
                    encoding_structure = constants.ENCD_KDTREE

                names = _fast_face_match_knn(known_encodings, encodings, known_names, constants.NORM_DIST_TOLERANCE,
                                             constants.K_NN)
            else:
                names = _linear_search(known_encodings, encodings, known_names)

        return names, boxes
