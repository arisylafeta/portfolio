---
date: "2023-10-01"
title: "Imaginify: Image Manipulation with Generative AI "
image: "images/post-4/cover.png"
categories: ["Data Science & AI", "Software Development"]
draft: false
---

#### Introduction

In recent years, interest in Artificial Intelligence (AI) has surged, marking a profound shift in the technological landscape. This growing fascination is not just a trend but a harbinger of a future where startups in the AI field are set to dominate, promising innovations that were once the domain of science fiction. As a simple project to dip my toes into this exciting field, I've built Imaginify. Imaginify is a Full-Stack Software-as-a-Service (SaaS) platform dedicated to revolutionizing the way we interact with images. Leveraging the power of Generative AI, Imaginify offers users an unparalleled experience in image manipulation. Built from the ground up using technologies like:
* NextJS 14
* Shadcn-UI
* TailwindCSS
* MongoDB

 this platform is designed with a modern UI and easy navigation for a great user experience. You can interact with the website by clicking the button below. Feel free to give it a try and manipulate some images!

 {{< button "Imaginify" "https://imaginify-lilac-nine.vercel.app/" >}} 

<hr>

#### User Authentication
Understanding the importance of security and personalization, Imaginify incorporates Clerk for robust user authentication. This ensures that visitors can register seamlessly while guaranteeing the safety of their data. Once registered, users' information is securely stored in a MongoDB database, offering a personalized experience. Each user is welcomed with their own collection of images and an initial allotment of coins, setting the stage for a custom-tailored interaction with the platform.

{{< image class="img-fluid rounded-100" title="image" src="/images/post-4/1.png" alt="element">}}
<hr>

#### Cloudinary Integration
At the heart of Imaginify's feature set is its integration with the Cloudinary API, enabling a suite of advanced image manipulation tools. Users can breathe new life into old photos with Image Restoration, unleash creativity with Generative Fill, effortlessly remove unwanted objects with Object Removal, add a splash of color through Object Recoloring, and seamlessly extract subjects with Background Removal. The platform offers the flexibility to upload personal images, find inspiration online, and provides the convenience of saving work both on the website and as local downloads. This comprehensive toolkit empowers users to transform their visions into reality without any technical constraints.

{{< image class="img-fluid rounded-6" title="image" src="/images/post-4/2.png" alt="element">}}
<hr>

#### Stripe Payment Gateway
Recognizing the value of accessibility, Imaginify welcomes new users with 15 free credits, allowing them to explore the platform's capabilities without initial investment. As users delve deeper and their needs grow, they can purchase additional credits through a straightforward and secure process facilitated by the Stripe Payment Gateway. This system ensures a smooth transactional experience, enabling users to focus on their creative endeavors without worrying about payment complications.

{{< image class="img-fluid rounded-6" title="image" src="/images/post-4/3.png" alt="element">}}

Hope you enjoy using it as much as I've enjoyed creating it.
<hr>