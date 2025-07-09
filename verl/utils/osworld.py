import copy


def remove_image_from_content(content):
    content_new = []
    for c in content:
        if c["type"] == "image":
            continue
        content_new.append(c)
    if len(content_new) == 0:
        return None
    return content_new


def limit_images_in_messages(msg, limit_images: int = 5) -> list:
    """
    Limit the number of images in a message to ensure it doesn't exceed limit_images.
    
    Args:
        msg: The message to process
        limit_images: Maximum number of images allowed in the message
    
    Returns:
        Processed message with limited images
    """
    msg = copy.deepcopy(msg)
    image_count = 0
    
    # Count total images in the message
    for m in msg:
        if not isinstance(m["content"], list):
            continue
        
        for c in m["content"]:
            if c["type"] == "image":
                image_count += 1
    
    # If image count is within limit, return original message
    if image_count <= limit_images:
        return msg
    
    # Process message to limit images
    current_image_count = 0
    for msg_idx in range(len(msg)-1, -1, -1):
        m = msg[msg_idx]
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c["type"] == "image":
                    current_image_count += 1
            if current_image_count > limit_images:
                m["content"] = remove_image_from_content(m["content"])
                msg[msg_idx] = m
    
    msg_for_prompt = []
    for m in msg:
        if isinstance(m["content"], list) and len(m["content"]) == 0:
            continue
        if m["content"] is None:
            continue
        msg_for_prompt.append(m)
    
    return msg_for_prompt