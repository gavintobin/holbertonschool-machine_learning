filtered_boxes, box_classes, box_scores = [], [], []

        for box, box_conf, box_class_prob in (zip(
                boxes, box_confidences, box_class_probs)):
            box_scores_per_class = box_conf * box_class_prob
            box_class = np.argmax(box_scores_per_class, axis=-1)
            box_score = np.max(box_scores_per_class, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.extend(box[mask])
            box_classes.extend(box_class[mask])
            box_scores.extend(box_score[mask])

        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores
